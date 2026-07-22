#!/usr/bin/env python3
"""Merge two ordered teacher exports and route disagreements for adjudication."""

import argparse
import io
import json
from pathlib import Path

import zstandard


def records(path):
    with Path(path).open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
            with io.TextIOWrapper(decoded, encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)


def boundaries(annotation):
    return [word for word, _pos in annotation or []]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--pos-disagreement", required=True)
    parser.add_argument("--hard", required=True)
    args = parser.parse_args()
    accepted = pos_disagreement = hard = 0
    with Path(args.accepted).open("w", encoding="utf-8") as good, Path(args.hard).open(
        "w", encoding="utf-8"
    ) as difficult, Path(args.pos_disagreement).open("w", encoding="utf-8") as pos_difficult:
        for left, right in zip(records(args.first), records(args.second), strict=True):
            if (left["sequence"], left["digest"]) != (right["sequence"], right["digest"]):
                raise ValueError("teacher exports are not aligned")
            left_annotation = left.get("annotation")
            right_annotation = right.get("annotation")
            same_boundary = boundaries(left_annotation) == boundaries(right_annotation)
            same_annotation = left_annotation == right_annotation
            item = {
                "digest": left["digest"], "text": left["text"],
                "ltp": left_annotation if "ltp" in left["teacher"] else right_annotation,
                "paddle": right_annotation if "paddle" in right["teacher"] else left_annotation,
                "same_boundary": same_boundary,
            }
            if same_annotation:
                good.write(json.dumps({**item, "annotation": left_annotation}, ensure_ascii=False) + "\n")
                accepted += 1
            elif same_boundary:
                pos_difficult.write(json.dumps(item, ensure_ascii=False) + "\n")
                pos_disagreement += 1
            else:
                difficult.write(json.dumps(item, ensure_ascii=False) + "\n")
                hard += 1
    print(json.dumps({
        "accepted": accepted, "pos_disagreement": pos_disagreement,
        "boundary_disagreement": hard,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
