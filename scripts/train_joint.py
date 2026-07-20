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
        positive_weight=metadata.get("positive_weight"),
        pu_target=metadata.get("pu_target", 0.0),
        pu_weight=metadata.get("pu_weight", 0.0),
    )


def train(args):
    if args.graph_max_span is None:
        args.graph_max_span = args.max_span
    if args.graph_max_span < 1 or args.graph_max_span > args.max_span:
        raise ValueError("--graph-max-span must be between 1 and --max-span")
    if not 0.0 <= args.pu_target <= 1.0:
        raise ValueError("--pu-target must be between 0 and 1")
    if args.pu_weight < 0.0:
        raise ValueError("--pu-weight must be non-negative")
    if args.graph_auxiliary_weight < 0.0:
        raise ValueError("--graph-auxiliary-weight must be non-negative")
    if not 0.0 <= args.graph_auxiliary_unlabelled_weight <= 1.0:
        raise ValueError("--graph-auxiliary-unlabelled-weight must be between 0 and 1")
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
            train_path, type_map, args.recognizer_batch_size, args.max_span, shuffle=True,
            gold_config=args.config if args.pu_weight > 0 else None, gold_mode=mode,
        )
        span_dev = SyntaxSpanSampleReader(
            dev_path, type_map, args.recognizer_batch_size, args.max_span,
            labels=span_train.labels, gold_config=args.config, gold_mode=mode
        )
    else:
        span_train = SpanSampleReader(
            train_path, args.recognizer_batch_size, args.max_span, shuffle=True,
            gold_config=args.config if args.pu_weight > 0 else None, gold_mode=mode,
        )
        span_dev = SpanSampleReader(
            dev_path, args.recognizer_batch_size, args.max_span,
            gold_config=args.config, gold_mode=mode,
        )
        negative_spans, positive_spans = span_train.class_counts()
        positive_weight = min(
            args.max_positive_weight,
            negative_spans / max(1, positive_spans),
        )
    graph_train = GraphSampleReader(
        train_path, args.config, mode, args.quantizer_batch_size,
        args.graph_max_span, shuffle=True, type_map=type_map
    )
    graph_dev = GraphSampleReader(
        dev_path, args.config, mode, args.quantizer_batch_size,
        args.graph_max_span, type_map=type_map
    )
    if span_train.wordsize() != graph_train.wordsize():
        raise RuntimeError("recognizer and quantizer must use the same WordPiece vocabulary")

    metadata = {
        "architecture": "joint-transformer-segmentation",
        "vocab_num": span_train.wordsize(),
        "hidden_size": args.hidden_size,
        "wtype_num": graph_train.typesize(),
        "max_span": args.max_span,
        "graph_max_span": args.graph_max_span,
        "recognizer_kind": args.recognizer_kind,
        "class_num": span_train.classsize() if args.recognizer_kind == "syntax" else None,
        "positive_weight": positive_weight if args.recognizer_kind == "binary" else None,
        "pu_target": args.pu_target,
        "pu_weight": args.pu_weight,
        "training_stage": args.training_stage,
        "labels": span_train.labels if args.recognizer_kind == "syntax" else None,
        "type_map": type_map,
        "graph_mode": mode,
        "train_data": train_path,
        "dev_data": dev_path,
        "quantizer_output": "association_negative_log_probability",
        "joint_update_mode": args.joint_update_mode,
        "quantizer_encoder_gradient": not args.graph_detach_context,
        "graph_auxiliary_weight": args.graph_auxiliary_weight,
        "graph_auxiliary_unlabelled_weight": args.graph_auxiliary_unlabelled_weight,
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
    model.set_training_stage(args.training_stage)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
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
        if args.training_stage == "quantizer":
            model.eval()
            model.graph_quantizer.quantizer.train()
        else:
            model.train()
        recognizer_losses = []
        quantizer_losses = []
        skipped_nonfinite = 0
        if args.training_stage == "recognizer":
            batches = ((span_batch, None) for span_batch in span_train)
        elif args.training_stage == "quantizer":
            batches = ((None, graph_batch) for graph_batch in graph_train)
        else:
            recognizer_iterator = iter(span_train)

            def joint_batches():
                nonlocal recognizer_iterator
                for graph_batch in graph_train:
                    try:
                        span_batch = next(recognizer_iterator)
                    except StopIteration:
                        recognizer_iterator = iter(span_train)
                        span_batch = next(recognizer_iterator)
                    if args.joint_update_mode == "alternating":
                        # Keep only one shared-encoder autograd graph alive per
                        # optimizer step. Both tasks still update the same
                        # parameters, but their heterogeneous backward graphs
                        # never merge in one CUDA engine traversal.
                        yield span_batch, None
                        yield None, graph_batch
                    else:
                        yield span_batch, graph_batch

            batches = joint_batches()
        for span_batch, graph_batch in batches:
            optimizer.zero_grad(set_to_none=True)
            recognizer_loss = None
            quantizer_loss = None
            if span_batch is not None:
                span_batch = tuple(tensor.to(device) for tensor in span_batch)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    recognizer_loss = model.recognizer(*span_batch)
            # GraphLossSparse repeatedly combines path log probabilities. Keep
            # the graph encoder and K/Q projections in FP32; FP16 autocast can
            # overflow their gradients on large candidate DAGs even when every
            # forward association NLL is finite.
            if graph_batch is not None:
                with torch.amp.autocast("cuda", enabled=False):
                    quantizer_loss = model.graph_quantizer(
                        *graph_batch,
                        detach_context=(
                            args.training_stage == "joint"
                            and args.graph_detach_context
                        ),
                        auxiliary_weight=args.graph_auxiliary_weight,
                        auxiliary_unlabelled_weight=args.graph_auxiliary_unlabelled_weight,
                    )
            if recognizer_loss is None:
                loss = (
                    args.quantizer_weight * quantizer_loss
                    if args.training_stage == "joint" else quantizer_loss
                )
            elif quantizer_loss is None:
                loss = recognizer_loss
            else:
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
            if recognizer_loss is not None:
                recognizer_losses.append(float(recognizer_loss.detach().cpu()))
            if quantizer_loss is not None:
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
        quantizer_dev_loss = (
            None if args.training_stage == "recognizer"
            else evaluate_quantizer(model.graph_quantizer, graph_dev)
        )
        scheduler.step()
        if args.training_stage == "recognizer":
            objective = recognizer_error
        elif args.training_stage == "quantizer":
            objective = quantizer_dev_loss
        else:
            objective = recognizer_error + args.quantizer_weight * quantizer_dev_loss
        metrics = {
            "epoch": epoch,
            "training_stage": args.training_stage,
            "recognizer_loss": (
                sum(recognizer_losses) / len(recognizer_losses)
                if recognizer_losses else None
            ),
            "quantizer_loss": (
                sum(quantizer_losses) / len(quantizer_losses)
                if quantizer_losses else None
            ),
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
            labels=metadata["labels"], gold_config=args.config, gold_mode=mode
        )
        recognizer_metrics = evaluate_syntax_recognizer(
            model.recognizer, spans, device,
            fixed_threshold=metadata.get("dev", {}).get("threshold"),
        )
    else:
        spans = SpanSampleReader(
            args.data, args.recognizer_batch_size, metadata["max_span"],
            gold_config=args.config, gold_mode=mode,
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
    command.add_argument(
        "--graph-max-span", type=int,
        help="negative candidate envelope for GraphLoss; defaults to --max-span",
    )
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--quantizer-weight", type=float, default=0.5)
    command.add_argument("--max-positive-weight", type=float, default=20.0,
                         help="cap the corpus-level binary positive-class weight")
    command.add_argument("--pu-target", type=float, default=0.0,
                         help="soft target for lexicon words absent from the annotated path")
    command.add_argument("--pu-weight", type=float, default=0.0,
                         help="loss weight for lexicon words absent from the annotated path")
    command.add_argument("--training-stage", choices=("joint", "recognizer", "quantizer"),
                         default="joint", help="joint training or one frozen staged task")
    command.add_argument(
        "--joint-update-mode", choices=("alternating", "combined"), default="alternating",
        help="alternate task optimizer steps or backpropagate both task graphs together",
    )
    command.add_argument(
        "--graph-auxiliary-weight", type=float, default=0.1,
        help="local edge loss weight used to train the encoder in alternating mode",
    )
    command.add_argument(
        "--graph-auxiliary-unlabelled-weight", type=float, default=0.05,
        help="PU weight for non-gold edges in the local encoder auxiliary loss",
    )
    command.add_argument(
        "--graph-detach-context", action=argparse.BooleanOptionalAction, default=False,
        help="detach GraphLoss from the encoder and use only the local auxiliary bridge",
    )
    command.add_argument("--clip-grad", type=float, default=1.0)
    command.add_argument("--resume", help="initialize model parameters from a compatible checkpoint")
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                         help="enable CUDA automatic mixed precision; use --no-amp for older GPUs")
    command.add_argument("--detect-anomaly", action="store_true")
    command.add_argument("--max-nonfinite-batches", type=int, default=8)
    command.add_argument("--seed", type=int, default=20260715)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(graph_max_span=None)
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
