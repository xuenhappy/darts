#!/usr/bin/env python3
"""Estimate smoothed label information and rewrite LabelEncoder hx files."""

import argparse
from collections import Counter
import math
from pathlib import Path


def labels_from_hx(path):
    return [
        line.split("#", 1)[0].strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if "#" in line and line.split("#", 1)[0].strip()
    ]


def count_dictionary(path, counts, allowed=None):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            labels, _word = line.split(":", 1)
            for label in labels.replace("<", "").replace(">", "").split(","):
                label = label.strip()
                if label and (allowed is None or label in allowed):
                    counts[label] += 1


def count_tagged_corpus(path, counts):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            for item in line.split():
                _word, separator, label = item.rpartition("/")
                if separator and label.startswith("POS_"):
                    counts[label] += 1


def write_information(labels, counts, output, smoothing):
    denominator = sum(counts.values()) + smoothing * len(labels)
    values = {}
    for label in labels:
        probability = (counts[label] + smoothing) / denominator
        values[label] = -math.log(probability)
    Path(output).write_text(
        "".join(f"{label}#{values[label]:.6f}\n" for label in labels),
        encoding="utf-8",
    )
    return values


def existing_information(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "#" not in line:
            continue
        label, value = line.split("#", 1)
        values[label.strip()] = float(value.strip())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type-hx", default="data/codes/type.hx.txt")
    parser.add_argument("--pos-hx", default="data/codes/pos.hx.txt")
    parser.add_argument("--dictionary", default="data/generated/dictionary.txt")
    parser.add_argument("--lac-corpus", default="data/generated/lac-train.txt")
    parser.add_argument("--lac-dictionary", default="data/demo/lac-dictionary.txt")
    parser.add_argument("--smoothing", type=float, default=1.0)
    args = parser.parse_args()
    if args.smoothing <= 0:
        parser.error("smoothing must be positive")

    type_labels = labels_from_hx(args.type_hx)
    type_counts = Counter()
    count_dictionary(args.dictionary, type_counts, set(type_labels))
    if type_counts:
        type_values = write_information(
            type_labels, type_counts, args.type_hx, args.smoothing
        )
    else:
        type_values = existing_information(args.type_hx)
        print("warning=no matching semantic labels; preserving type hx information")

    pos_labels = labels_from_hx(args.pos_hx)
    pos_counts = Counter()
    count_tagged_corpus(args.lac_corpus, pos_counts)
    count_dictionary(args.lac_dictionary, pos_counts, set(pos_labels))
    pos_values = write_information(
        pos_labels, pos_counts, args.pos_hx, args.smoothing
    )
    print(
        f"type_labels={len(type_values)} pos_labels={len(pos_values)} "
        f"type_events={sum(type_counts.values())} pos_events={sum(pos_counts.values())}"
    )


if __name__ == "__main__":
    main()
