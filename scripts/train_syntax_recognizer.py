#!/usr/bin/env python3
"""Train a NOT_WORD plus POS multi-class span recognizer."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from darts.devel.model import SyntaxSpanRecognizer
from darts.devel.reader import SyntaxSpanSampleReader
from darts.devel.utils import stable_clip_grad_norm_
from train_recognizer import checkpoint, select_device


@torch.no_grad()
def evaluate(model, reader, device):
    model.eval()
    correct = total = word_correct = word_total = 0
    for word_ids, lengths, spans in reader:
        word_ids, lengths, spans = word_ids.to(device), lengths.to(device), spans.to(device)
        predicted = model.logits(word_ids, lengths, spans[:, :3]).argmax(dim=-1)
        gold = spans[:, -1]
        correct += int((predicted == gold).sum())
        total += gold.numel()
        mask = gold != 0
        word_correct += int(((predicted == gold) & mask).sum())
        word_total += int(mask.sum())
    return {
        "accuracy": correct / total if total else 0.0,
        "word_type_accuracy": word_correct / word_total if word_total else 0.0,
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_reader = SyntaxSpanSampleReader(args.train, args.type_map, args.batch_size,
                                          args.max_span, shuffle=True)
    dev_reader = SyntaxSpanSampleReader(args.dev, args.type_map, args.batch_size, args.max_span)
    metadata = {
        "architecture": "transformer-syntax-span-recognizer",
        "vocab_num": train_reader.wordsize(),
        "hidden_size": args.hidden_size,
        "class_num": train_reader.classsize(),
        "labels": train_reader.labels,
        "max_span": args.max_span,
        "output": "not_word_plus_pos_probability",
        "seed": args.seed,
    }
    model = SyntaxSpanRecognizer(metadata["vocab_num"], args.hidden_size,
                                 metadata["class_num"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    output = Path(args.output_dir)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for word_ids, lengths, spans in train_reader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(word_ids.to(device), lengths.to(device), spans.to(device))
            loss.backward()
            stable_clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, dev_reader, device)
        print(json.dumps({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), **metrics}))
        checkpoint(model, {**metadata, "dev": metrics}, output / "last.pt")
        if metrics["word_type_accuracy"] > best:
            best = metrics["word_type_accuracy"]
            checkpoint(model, {**metadata, "dev": metrics}, output / "best.pt")


def export(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    model = SyntaxSpanRecognizer(metadata["vocab_num"], metadata["hidden_size"],
                                 metadata["class_num"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.export2onnx(str(output))
    output.with_suffix(".labels.txt").write_text(
        "\n".join(metadata["labels"]) + "\n", encoding="utf-8"
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--train", default="data/generated/lac-train.txt")
    command.add_argument("--dev", default="data/generated/lac-dev.txt")
    command.add_argument("--type-map", default="data/codes/pos.hx.txt")
    command.add_argument("--output-dir", default="model_bin/syntax")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--weight-decay", type=float, default=1e-2)
    command.add_argument("--clip-grad", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=20260716)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=train)
    command = sub.add_parser("export")
    command.add_argument("checkpoint")
    command.add_argument("output")
    command.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
