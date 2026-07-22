#!/usr/bin/env python3
"""Evaluate an external LAC teacher against manually annotated Darts corpora."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from pseudo_corpus import make_teacher


def read_gold(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if not fields:
                continue
            sentence = []
            for field in fields:
                word, separator, pos = field.rpartition("/")
                if not separator:
                    raise ValueError(f"invalid LAC field: {field!r}")
                sentence.append((word, pos))
            yield sentence


def spans(annotation):
    offset = 0
    result = []
    for word, pos in annotation:
        end = offset + len(word)
        result.append((offset, end, pos))
        offset = end
    return result


def benchmark(args):
    gold = list(read_gold(args.gold))
    teacher = make_teacher(args.teacher, args.batch_size, args.device_id, args.model)
    boundary_tp = predicted = expected = typed_correct = aligned = 0
    pos_gold = Counter()
    pos_correct = Counter()
    started = time.perf_counter()
    for start in range(0, len(gold), args.batch_size):
        batch = gold[start:start + args.batch_size]
        texts = ["".join(word for word, _ in sentence) for sentence in batch]
        outputs = teacher.annotate(texts)
        for text, target, output in zip(texts, batch, outputs):
            if "".join(word for word, _ in output) != text:
                continue
            aligned += 1
            target_spans = {(left, right): pos for left, right, pos in spans(target)}
            output_spans = {(left, right): pos for left, right, pos in spans(output)}
            shared = target_spans.keys() & output_spans.keys()
            boundary_tp += len(shared)
            predicted += len(output_spans)
            expected += len(target_spans)
            for span in shared:
                gold_pos = target_spans[span]
                pos_gold[gold_pos] += 1
                if output_spans[span] == gold_pos:
                    typed_correct += 1
                    pos_correct[gold_pos] += 1
    precision = boundary_tp / max(1, predicted)
    recall = boundary_tp / max(1, expected)
    metrics = {
        "teacher": teacher.name,
        "gold": args.gold,
        "sentences": len(gold),
        "aligned_sentences": aligned,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "pos_accuracy_on_aligned_words": typed_correct / max(1, boundary_tp),
        "sentences_per_second": len(gold) / max(1e-9, time.perf_counter() - started),
        "pos_recall": {
            pos: pos_correct[pos] / count for pos, count in pos_gold.most_common()
        },
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/generated/lac-dev.txt")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--teacher", choices=("paddle", "ltp"), default="paddle")
    parser.add_argument("--model", default="LTP/small", help="LTP model id; ignored by Paddle")
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
