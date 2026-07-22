#!/usr/bin/env python3
"""Export lossless external-teacher annotations for later consensus filtering."""

import argparse
import json
from pathlib import Path

import zstandard

from pseudo_corpus import digest, input_texts, make_teacher, split_long, validate_annotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", choices=("ltp", "paddle"), required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-atoms", type=int, default=108)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model", default="LTP/small", help="LTP model id; ignored by Paddle")
    args = parser.parse_args()

    teacher = make_teacher(args.teacher, args.batch_size, args.device_id, args.model)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=6)
    batch = []
    sequence = 0
    with target.open("wb") as raw, compressor.stream_writer(raw) as output:
        def consume():
            nonlocal sequence
            for text, annotation in zip(batch, teacher.annotate(batch)):
                reason = validate_annotation(text, annotation, 1, args.max_atoms)
                record = {
                    "sequence": sequence,
                    "digest": digest(text).hex(),
                    "text": text,
                    "teacher": teacher.name,
                    "annotation": annotation if reason is None else None,
                    "error": reason,
                }
                output.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                sequence += 1

        for _path, _line, raw_text in input_texts(args.inputs, args.text_field):
            for sentence in split_long(raw_text, args.max_atoms):
                batch.append(sentence)
                if len(batch) == args.batch_size:
                    consume()
                    batch.clear()
        if batch:
            consume()


if __name__ == "__main__":
    main()
