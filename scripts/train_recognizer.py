#!/usr/bin/env python3
"""Train and export the overlapping-span Transformer recognizer."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from darts.devel.model import SpanRecognizer
from darts.devel.reader import SpanSampleReader


@torch.no_grad()
def evaluate(model, reader, device, fixed_thresholds=None):
    """Evaluate spans, calibrating thresholds only when none are supplied."""
    model.eval()
    probabilities = []
    labels = []
    span_lengths = []
    for word_ids, lengths, span_info in reader:
        word_ids = word_ids.to(device)
        lengths = lengths.to(device)
        span_info = span_info.to(device)
        probabilities.append(torch.sigmoid(model.logits(word_ids, lengths, span_info[:, :3])).cpu())
        span_lengths.append(span_info[:, -2].cpu())
        labels.append(span_info[:, -1].bool().cpu())
    if not probabilities:
        raise RuntimeError(f"no recognizer samples were generated from {reader.sample}")
    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)
    span_lengths = torch.cat(span_lengths)
    thresholds = dict(fixed_thresholds or {})
    if fixed_thresholds is None:
        # Longer spans have many more negative combinations and usually need a
        # stricter threshold. Select each length only on development data.
        for length in sorted(span_lengths.unique().tolist()):
            mask = span_lengths == length
            if not labels[mask].any():
                thresholds[str(length)] = 0.95
                continue
            best_length = None
            for threshold in np.arange(0.10, 0.96, 0.05):
                predicted = probabilities[mask] >= threshold
                gold = labels[mask]
                true_positive = int((predicted & gold).sum())
                predicted_total = int(predicted.sum())
                gold_total = int(gold.sum())
                precision = true_positive / predicted_total if predicted_total else 0.0
                recall = true_positive / gold_total if gold_total else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                if best_length is None or f1 > best_length["f1"] or (
                        f1 == best_length["f1"] and threshold > best_length["threshold"]):
                    best_length = {"threshold": round(float(threshold), 2), "f1": f1}
            thresholds[str(length)] = best_length["threshold"]
    missing = sorted({str(int(length)) for length in span_lengths} - thresholds.keys())
    if missing:
        raise RuntimeError(f"missing fixed recognizer thresholds for atom lengths {missing}")
    selected = torch.tensor([thresholds[str(int(length))] for length in span_lengths])
    predicted = probabilities >= selected
    true_positive = int((predicted & labels).sum())
    predicted_total = int(predicted.sum())
    gold_total = int(labels.sum())
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"thresholds": thresholds, "precision": precision, "recall": recall, "f1": f1}


def checkpoint(model, metadata, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def select_device(requested):
    selected = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    selected = "cpu" if selected == "auto" else selected
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(selected)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = select_device(args.device)
    train_reader = SpanSampleReader(args.train, batch_size=args.batch_size, max_span=args.max_span,
                                    shuffle=True)
    dev_reader = SpanSampleReader(args.dev, batch_size=args.batch_size, max_span=args.max_span)
    metadata = {"vocab_num": train_reader.wordsize(), "hidden_size": args.hidden_size,
                "max_span": args.max_span, "architecture": "transformer-span-recognizer",
                "output": "word_probability", "seed": args.seed}
    model = SpanRecognizer(metadata["vocab_num"], metadata["hidden_size"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output_dir)
    best_f1 = -1.0
    stale = 0
    if not any(True for _ in train_reader):
        raise RuntimeError(f"no recognizer samples were generated from {args.train}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for word_ids, lengths, span_info in train_reader:
            optimizer.zero_grad(set_to_none=True)
            word_ids = word_ids.to(device)
            lengths = lengths.to(device)
            span_info = span_info.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(word_ids, lengths, span_info)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, dev_reader, device)
        scheduler.step()
        print(json.dumps({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)),
                          "learning_rate": scheduler.get_last_lr()[0], **metrics}))
        checkpoint(model, {**metadata, "dev": metrics}, output / "last.pt")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            stale = 0
            checkpoint(model, {**metadata, "dev": metrics}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"best_checkpoint={output / 'best.pt'} best_dev_f1={best_f1:.6f}")


def export(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = SpanRecognizer(metadata["vocab_num"], metadata["hidden_size"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.export2onnx(str(output))
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(f"exported={output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train")
    command.add_argument("--train", default="data/generated/cws-train.txt")
    command.add_argument("--dev", default="data/generated/cws-dev.txt")
    command.add_argument("--output-dir", default="model_bin/recognizer")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--patience", type=int, default=4)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--clip-grad", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=20260715)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=train)
    command = commands.add_parser("export")
    command.add_argument("checkpoint")
    command.add_argument("output")
    command.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
