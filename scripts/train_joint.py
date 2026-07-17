#!/usr/bin/env python3
"""Jointly train one word encoder for recognizer and graph quantizer tasks."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from darts.devel.model import JointSegmentationTrainer
from darts.devel.reader import GraphSampleReader, SpanSampleReader, SyntaxSpanSampleReader
from darts.devel.utils import stable_clip_grad_norm_
from train_quantizer import evaluate as evaluate_quantizer
from train_recognizer import evaluate as evaluate_recognizer, select_device
from train_syntax_recognizer import evaluate as evaluate_syntax_recognizer


def save(model, metadata, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def build_model(metadata):
    return JointSegmentationTrainer(
        metadata["vocab_num"],
        metadata["hidden_size"],
        metadata["wtype_num"],
        recognizer_kind=metadata.get("recognizer_kind", "binary"),
        class_num=metadata.get("class_num"),
    )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        # PyTorch's fused efficient SDPA backward can return NaN on some
        # consumer/Ampere GPUs for heavily padded variable-length batches.
        # The math kernel is slower but stable and does not change the model
        # architecture or exported ONNX graph.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    device = select_device(args.device)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    train_path = args.train or (
        "data/generated/lac-train.txt"
        if args.recognizer_kind == "syntax"
        else "data/generated/cws-train.txt"
    )
    dev_path = args.dev or (
        "data/generated/lac-dev.txt"
        if args.recognizer_kind == "syntax"
        else "data/generated/cws-dev.txt"
    )
    mode = args.mode or ("lac" if args.recognizer_kind == "syntax" else "hybrid")
    type_map = args.type_map or (
        "data/codes/pos.hx.txt"
        if args.recognizer_kind == "syntax"
        else "data/codes/type.hx.txt"
    )
    if args.recognizer_kind == "syntax":
        span_train = SyntaxSpanSampleReader(
            train_path, type_map, args.recognizer_batch_size, args.max_span, shuffle=True
        )
        span_dev = SyntaxSpanSampleReader(
            dev_path, type_map, args.recognizer_batch_size, args.max_span,
            labels=span_train.labels
        )
    else:
        span_train = SpanSampleReader(
            train_path, args.recognizer_batch_size, args.max_span, shuffle=True
        )
        span_dev = SpanSampleReader(dev_path, args.recognizer_batch_size, args.max_span)
    graph_train = GraphSampleReader(
        train_path, args.config, mode, args.quantizer_batch_size,
        args.max_span, shuffle=True, type_map=type_map
    )
    graph_dev = GraphSampleReader(
        dev_path, args.config, mode, args.quantizer_batch_size,
        args.max_span, type_map=type_map
    )
    if span_train.wordsize() != graph_train.wordsize():
        raise RuntimeError("recognizer and quantizer must use the same WordPiece vocabulary")

    metadata = {
        "architecture": "joint-transformer-segmentation",
        "vocab_num": span_train.wordsize(),
        "hidden_size": args.hidden_size,
        "wtype_num": graph_train.typesize(),
        "max_span": args.max_span,
        "recognizer_kind": args.recognizer_kind,
        "class_num": span_train.classsize() if args.recognizer_kind == "syntax" else None,
        "labels": span_train.labels if args.recognizer_kind == "syntax" else None,
        "type_map": type_map,
        "graph_mode": mode,
        "train_data": train_path,
        "dev_data": dev_path,
        "quantizer_output": "association_negative_log_probability",
        "seed": args.seed,
    }
    model = build_model(metadata).to(device)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        resumed_metadata = resumed["metadata"]
        for key in ("vocab_num", "hidden_size", "wtype_num", "max_span"):
            if resumed_metadata[key] != metadata[key]:
                raise RuntimeError(f"resume checkpoint {key} does not match current training data")
        if resumed_metadata.get("recognizer_kind", "binary") != args.recognizer_kind:
            raise RuntimeError("resume checkpoint recognizer kind does not match")
        if resumed_metadata.get("class_num") != metadata["class_num"]:
            raise RuntimeError("resume checkpoint class count does not match")
        model.load_state_dict(resumed["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output = Path(args.output_dir)
    best_objective = float("inf")
    stale = 0
    if not any(True for _ in span_train):
        raise RuntimeError(f"no recognizer samples were generated from {train_path}")
    if not any(True for _ in graph_train):
        raise RuntimeError(f"no quantizer samples were generated from {train_path}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        recognizer_iterator = iter(span_train)
        recognizer_losses = []
        quantizer_losses = []
        skipped_nonfinite = 0
        # The graph task determines epoch length. Restarting the cheaper span
        # iterator balances task update counts without retaining yielded batches.
        for graph_batch in graph_train:
            try:
                span_batch = next(recognizer_iterator)
            except StopIteration:
                recognizer_iterator = iter(span_train)
                span_batch = next(recognizer_iterator)
            optimizer.zero_grad(set_to_none=True)
            span_batch = tuple(tensor.to(device) for tensor in span_batch)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                recognizer_loss = model.recognizer(*span_batch)
            # GraphLossSparse repeatedly combines path log probabilities. Keep
            # the graph encoder and K/Q projections in FP32; FP16 autocast can
            # overflow their gradients on large candidate DAGs even when every
            # forward association NLL is finite.
            with torch.amp.autocast("cuda", enabled=False):
                quantizer_loss = model.graph_quantizer(*graph_batch)
            loss = recognizer_loss + args.quantizer_weight * quantizer_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_gradients = [name for name, parameter in model.named_parameters()
                             if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if bad_gradients:
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite += 1
                print(json.dumps({
                    "warning": "skipped_nonfinite_batch",
                    "epoch": epoch,
                    "count": skipped_nonfinite,
                    "parameters": bad_gradients,
                }))
                if skipped_nonfinite > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"too many non-finite batches in epoch {epoch}: {skipped_nonfinite}"
                    )
                continue
            stable_clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            recognizer_losses.append(float(recognizer_loss.detach().cpu()))
            quantizer_losses.append(float(quantizer_loss.detach().cpu()))

        if args.recognizer_kind == "syntax":
            recognizer_metrics = evaluate_syntax_recognizer(model.recognizer, span_dev, device)
            recognizer_error = (
                1.0 - recognizer_metrics["word_f1"] +
                0.25 * (1.0 - recognizer_metrics["word_type_accuracy"])
            )
        else:
            recognizer_metrics = evaluate_recognizer(model.recognizer, span_dev, device)
            recognizer_error = 1.0 - recognizer_metrics["f1"]
        quantizer_dev_loss = evaluate_quantizer(model.graph_quantizer, graph_dev)
        scheduler.step()
        objective = recognizer_error + args.quantizer_weight * quantizer_dev_loss
        metrics = {
            "epoch": epoch,
            "recognizer_loss": sum(recognizer_losses) / max(1, len(recognizer_losses)),
            "quantizer_loss": sum(quantizer_losses) / max(1, len(quantizer_losses)),
            "quantizer_dev_loss": quantizer_dev_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_mb": (torch.cuda.max_memory_allocated(device) / 1024**2
                            if device.type == "cuda" else 0.0),
            "skipped_nonfinite_batches": skipped_nonfinite,
            **recognizer_metrics,
        }
        print(json.dumps(metrics))
        epoch_metadata = {**metadata, "dev": metrics}
        save(model, epoch_metadata, output / "last.pt")
        if objective < best_objective:
            best_objective = objective
            stale = 0
            save(model, epoch_metadata, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"best_checkpoint={output / 'best.pt'} best_objective={best_objective:.6f}")


def export(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = build_model(metadata)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    syntax = metadata.get("recognizer_kind", "binary") == "syntax"
    recognizer_name = "syntax.onnx" if syntax else "recognizer.onnx"
    indicator_name = "lac-indicator.onnx" if syntax else "indicator.onnx"
    quantizer_name = "lac-quantizer.onnx" if syntax else "quantizer.onnx"
    model.recognizer.export2onnx(str(output / recognizer_name))
    model.encoder.export2onnx(str(output / indicator_name))
    model.graph_quantizer.quantizer.export2onnx(str(output / quantizer_name))
    if syntax:
        (output / "syntax.labels.txt").write_text(
            "\n".join(metadata["labels"]) + "\n", encoding="utf-8"
        )
    thresholds = metadata.get("dev", {}).get("thresholds", {})
    runtime = {
        "model.path": recognizer_name,
        "pmodel.path": indicator_name,
        "qmodel.path": quantizer_name,
        "max.span": str(metadata["max_span"]),
        "thresholds": {str(length): str(value) for length, value in thresholds.items()},
    }
    if syntax:
        runtime["label.path"] = "syntax.labels.txt"
        runtime["threshold"] = str(metadata.get("dev", {}).get("threshold", 0.5))
    metadata_name = "lac.json" if syntax else "neural.json"
    (output / metadata_name).write_text(
        json.dumps({**metadata, "runtime": runtime}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"models={output}")


def evaluate(args):
    """Evaluate one checkpoint without updating it or selecting on test data."""
    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = build_model(metadata)
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    syntax = metadata.get("recognizer_kind", "binary") == "syntax"
    mode = args.mode or metadata.get("graph_mode", "lac" if syntax else "hybrid")
    type_map = args.type_map or metadata.get(
        "type_map", "data/codes/pos.hx.txt" if syntax else "data/codes/type.hx.txt"
    )
    if syntax:
        spans = SyntaxSpanSampleReader(
            args.data, type_map, args.recognizer_batch_size, metadata["max_span"],
            labels=metadata["labels"]
        )
        recognizer_metrics = evaluate_syntax_recognizer(
            model.recognizer, spans, device,
            fixed_threshold=metadata.get("dev", {}).get("threshold"),
        )
    else:
        spans = SpanSampleReader(
            args.data, args.recognizer_batch_size, metadata["max_span"]
        )
        fixed_thresholds = metadata.get("dev", {}).get("thresholds")
        if not fixed_thresholds:
            raise RuntimeError("checkpoint has no development-set recognizer thresholds")
        recognizer_metrics = evaluate_recognizer(
            model.recognizer, spans, device, fixed_thresholds
        )
    graphs = GraphSampleReader(
        args.data, args.config, mode, args.quantizer_batch_size,
        metadata["max_span"], type_map=type_map
    )
    quantizer_loss = evaluate_quantizer(model.graph_quantizer, graphs)
    print(json.dumps({"data": args.data, "quantizer_loss": quantizer_loss,
                      **recognizer_metrics}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train")
    command.add_argument("--train")
    command.add_argument("--dev")
    command.add_argument("--config", default="data/conf.json")
    command.add_argument("--mode")
    command.add_argument("--type-map")
    command.add_argument("--recognizer-kind", choices=("binary", "syntax"), default="binary")
    command.add_argument("--output-dir", default="model_bin/joint")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--patience", type=int, default=4)
    command.add_argument("--recognizer-batch-size", type=int, default=64)
    command.add_argument("--quantizer-batch-size", type=int, default=16)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--quantizer-weight", type=float, default=0.5)
    command.add_argument("--clip-grad", type=float, default=1.0)
    command.add_argument("--resume", help="initialize model parameters from a compatible checkpoint")
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                         help="enable CUDA automatic mixed precision; use --no-amp for older GPUs")
    command.add_argument("--detect-anomaly", action="store_true")
    command.add_argument("--max-nonfinite-batches", type=int, default=8)
    command.add_argument("--seed", type=int, default=20260715)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=train)
    command = commands.add_parser("export")
    command.add_argument("checkpoint")
    command.add_argument("output_dir")
    command.set_defaults(func=export)
    command = commands.add_parser("evaluate")
    command.add_argument("checkpoint")
    command.add_argument("--data", default="data/generated/cws-test.txt")
    command.add_argument("--config", default="data/conf.json")
    command.add_argument("--mode")
    command.add_argument("--type-map")
    command.add_argument("--recognizer-batch-size", type=int, default=64)
    command.add_argument("--quantizer-batch-size", type=int, default=16)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
