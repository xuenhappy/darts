#!/usr/bin/env python3
"""Train and export the Transformer candidate-graph association quantizer.

The exported quantizer value is -log P(next word is associated with previous
word). GraphLossSparse normalizes complete paths rather than treating edges as
independent binary examples.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from darts.devel.model import GraphQuantizerTrainer
from darts.devel.reader import GraphSampleReader
from darts.devel.utils import stable_clip_grad_norm_


@torch.no_grad()
def evaluate(model, reader):
    model.eval()
    losses = [float(model(*batch).detach().cpu()) for batch in reader]
    if not losses:
        raise RuntimeError(f"no quantizer samples were generated from {reader.sample}")
    return sum(losses) / len(losses)


def save(model, metadata, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    selected = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if selected == "auto":
        selected = "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    train_reader = GraphSampleReader(args.train, args.config, args.mode, args.batch_size, shuffle=True)
    dev_reader = GraphSampleReader(args.dev, args.config, args.mode, args.batch_size)
    metadata = {"vocab_num": train_reader.wordsize(), "hidden_size": args.hidden_size,
                "wtype_num": train_reader.typesize(), "architecture": "transformer-graph-quantizer",
                "seed": args.seed}
    model = GraphQuantizerTrainer(metadata["vocab_num"], metadata["hidden_size"],
                                  metadata["wtype_num"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output_dir)
    best_loss = float("inf")
    stale = 0
    if not any(True for _ in train_reader):
        raise RuntimeError(f"no quantizer samples were generated from {args.train}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_reader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model(*batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            stable_clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        dev_loss = evaluate(model, dev_reader)
        scheduler.step()
        print(json.dumps({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)),
                          "dev_loss": dev_loss, "learning_rate": scheduler.get_last_lr()[0]}))
        save(model, metadata, output / "last.pt")
        if dev_loss < best_loss:
            best_loss = dev_loss
            stale = 0
            save(model, {**metadata, "dev_loss": dev_loss}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"best_checkpoint={output / 'best.pt'} best_dev_loss={best_loss:.6f}")


def export(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = GraphQuantizerTrainer(metadata["vocab_num"], metadata["hidden_size"], metadata["wtype_num"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.predictor.export2onnx(str(output / "indicator.onnx"))
    model.quantizer.export2onnx(str(output / "quantizer.onnx"))
    (output / "quantizer.json").write_text(
        json.dumps({**metadata, "output": "association_negative_log_probability"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"indicator={output / 'indicator.onnx'}\nquantizer={output / 'quantizer.onnx'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train")
    command.add_argument("--train", default="data/generated/cws-train.txt")
    command.add_argument("--dev", default="data/generated/cws-dev.txt")
    command.add_argument("--config", default="data/conf.json")
    command.add_argument("--mode", default="hybrid")
    command.add_argument("--output-dir", default="model_bin/quantizer")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--patience", type=int, default=4)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=2e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--clip-grad", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=20260715)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=train)
    command = commands.add_parser("export")
    command.add_argument("checkpoint")
    command.add_argument("output_dir")
    command.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
