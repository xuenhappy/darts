#!/usr/bin/env python3
"""Build gold-scored LTP/Paddle routing data for small-LLM LoRA training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from llm_adjudicate_lac import prompt
from pseudo_corpus import digest, POS_TO_SHORT


def read_gold(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        annotation = []
        for field in line.split():
            word, separator, pos = field.rpartition("/")
            if not separator:
                raise ValueError(f"invalid gold token: {field!r}")
            annotation.append((word, pos))
        if annotation:
            text = "".join(word for word, _pos in annotation)
            yield digest(text).hex(), text, annotation


def spans(annotation):
    offset = 0
    result = {}
    for word, pos in annotation:
        end = offset + len(word)
        result[(offset, end)] = pos
        offset = end
    return result


def score(gold, candidate):
    target = spans(gold)
    predicted = spans(candidate)
    shared = target.keys() & predicted.keys()
    precision = len(shared) / max(1, len(predicted))
    recall = len(shared) / max(1, len(target))
    boundary_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    typed = sum(target[key] == predicted[key] for key in shared) / max(1, len(target))
    return boundary_f1, 0.8 * boundary_f1 + 0.2 * typed


def export_raw(args):
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for key, text, _annotation in read_gold(args.gold):
            output.write(json.dumps({"digest": key, "text": text}, ensure_ascii=False) + "\n")


def flat(annotation):
    return " ".join(f"{word}/{POS_TO_SHORT[pos]}" for word, pos in annotation)


def correction_prompt(text, annotation):
    return (
        "/no_think\n纠正下面的LTP中文分词和词性标注。必须逐字覆盖原文，不增删字符。\n"
        "输出格式只能是空白分隔的 词/POS 标签，不要JSON，不要解释。\n"
        f"原文：{text}\nLTP标注：{flat(annotation)}"
    )


def build_correction(args):
    from teacher_consensus import records
    random.seed(args.seed)
    gold = {key: (text, annotation) for key, text, annotation in read_gold(args.gold)}
    ltp = {item["digest"]: item.get("annotation") for item in records(args.ltp)}
    counts = Counter()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for key, (text, target_annotation) in gold.items():
            source = ltp.get(key)
            if not source:
                counts["teacher_error"] += 1
                continue
            changed = source != target_annotation
            if not changed and random.random() >= args.identity_ratio:
                counts["identity_skipped"] += 1
                continue
            record = {
                "messages": [
                    {"role": "user", "content": correction_prompt(text, source)},
                    {"role": "assistant", "content": flat(target_annotation)},
                ],
                "digest": key, "text": text, "ltp": source,
                "gold": target_annotation, "changed": changed,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["changed" if changed else "identity"] += 1
    print(json.dumps({"gold": len(gold), "counts": counts}, ensure_ascii=False, default=dict))


def build(args):
    from teacher_consensus import records
    gold = {key: (text, annotation) for key, text, annotation in read_gold(args.gold)}
    teachers = {}
    for name, path in (("ltp", args.ltp), ("paddle", args.paddle)):
        teachers[name] = {item["digest"]: item.get("annotation") for item in records(path)}
    counts = Counter()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as output:
        for key, (text, target_annotation) in gold.items():
            ltp, paddle = teachers["ltp"].get(key), teachers["paddle"].get(key)
            if not ltp or not paddle:
                counts["teacher_error"] += 1
                continue
            if ltp == paddle:
                counts["teacher_agreement"] += 1
                continue
            ltp_boundary, ltp_score = score(target_annotation, ltp)
            paddle_boundary, paddle_score = score(target_annotation, paddle)
            if max(ltp_boundary, paddle_boundary) < args.reject_boundary_f1:
                choice = "reject"
            elif abs(ltp_score - paddle_score) < args.min_margin:
                counts["ambiguous"] += 1
                continue
            else:
                choice = "ltp" if ltp_score > paddle_score else "paddle"
            item = {"text": text, "ltp": ltp, "paddle": paddle}
            record = {
                "messages": [
                    {"role": "user", "content": prompt(item)},
                    {"role": "assistant", "content": json.dumps({"choice": choice})},
                ],
                "digest": key, "choice": choice,
                "ltp_boundary_f1": ltp_boundary,
                "paddle_boundary_f1": paddle_boundary,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[choice] += 1
    print(json.dumps({"gold": len(gold), "output": sum(counts[c] for c in ("ltp", "paddle", "reject")),
                      "counts": counts}, ensure_ascii=False, default=dict))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("raw")
    raw.add_argument("--gold", required=True)
    raw.add_argument("--output", required=True)
    raw.set_defaults(func=export_raw)
    dataset = subparsers.add_parser("build")
    dataset.add_argument("--gold", required=True)
    dataset.add_argument("--ltp", required=True)
    dataset.add_argument("--paddle", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--reject-boundary-f1", type=float, default=0.65)
    dataset.add_argument("--min-margin", type=float, default=0.02)
    dataset.set_defaults(func=build)
    correction = subparsers.add_parser("correct")
    correction.add_argument("--gold", required=True)
    correction.add_argument("--ltp", required=True)
    correction.add_argument("--output", required=True)
    correction.add_argument("--identity-ratio", type=float, default=0.2)
    correction.add_argument("--seed", type=int, default=17)
    correction.set_defaults(func=build_correction)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
