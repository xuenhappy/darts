#!/usr/bin/env python3
"""Compare teacher and consensus-choice boundaries against a gold CWS corpus."""

import argparse
import json
from pathlib import Path

from pseudo_corpus import digest
from teacher_consensus import records


def span_set(words):
    offset = 0
    result = set()
    for word in words:
        end = offset + len(word)
        result.add((offset, end))
        offset = end
    return result


def metrics(gold, predicted):
    tp = expected = actual = 0
    for key, words in gold.items():
        target = span_set(words)
        output = span_set([word for word, _pos in predicted[key]])
        tp += len(target & output)
        expected += len(target)
        actual += len(output)
    precision = tp / max(1, actual)
    recall = tp / max(1, expected)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def jsonl_annotations(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            yield item["digest"], item["annotation"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--ltp", required=True)
    parser.add_argument("--paddle", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--pos-disagreement", required=True)
    parser.add_argument("--llm-choice", required=True)
    args = parser.parse_args()
    gold = {}
    with Path(args.gold).open(encoding="utf-8") as stream:
        for line in stream:
            words = line.strip().split()
            if words:
                gold[digest("".join(words)).hex()] = words
    teacher_maps = {}
    for name, path in (("ltp", args.ltp), ("paddle", args.paddle)):
        teacher_maps[name] = {
            item["digest"]: item["annotation"] for item in records(path)
            if item.get("annotation") is not None and item["digest"] in gold
        }
    combined = dict(jsonl_annotations(args.accepted))
    # POS-only disagreement has identical boundaries; LTP is the stronger boundary teacher.
    with Path(args.pos_disagreement).open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            combined[item["digest"]] = item["ltp"]
    combined.update(jsonl_annotations(args.llm_choice))
    if combined.keys() != gold.keys():
        raise ValueError(f"coverage mismatch: gold={len(gold)} combined={len(combined)}")
    result = {name: metrics(gold, predictions) for name, predictions in teacher_maps.items()}
    result["consensus_llm"] = metrics(gold, combined)
    result["sentences"] = len(gold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
