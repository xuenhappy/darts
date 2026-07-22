#!/usr/bin/env python3
"""Build large pseudo-labelled CWS/LAC corpora with an external teacher."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid


SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|\n+")
SAFE_BREAK = re.compile(r"(?<=[，,、：:])")
ATOM = re.compile(r"[A-Za-z]+(?:[-_.'][A-Za-z0-9]+)*|\d+(?:[.:/%-]\d+)*|[^\s]", re.UNICODE)


POS_MAP = {
    "PER": "POS_nr", "LOC": "POS_ns", "ORG": "POS_nt",
    "TIME": "POS_t", "nr": "POS_nr", "ns": "POS_ns", "nt": "POS_nt",
    "nz": "POS_nz", "n": "POS_NOUN", "v": "POS_VERB", "vd": "POS_VERB",
    "vn": "POS_VERB", "a": "POS_ADJ", "ad": "POS_ADV", "d": "POS_ADV",
    "r": "POS_PRON", "p": "POS_ADP", "c": "POS_CCONJ", "u": "POS_PART",
    "m": "POS_NUM", "q": "POS_NUM", "f": "POS_f", "w": "POS_PUNCT",
    "xc": "POS_X", "s": "POS_NOUN", "t": "POS_t",
    # LTP4 863-style extensions.
    "nh": "POS_nr", "ni": "POS_nt", "nl": "POS_NOUN", "nd": "POS_f",
    "wp": "POS_PUNCT", "ws": "POS_X", "j": "POS_nz", "i": "POS_nz",
    "b": "POS_ADJ", "z": "POS_ADJ", "e": "POS_X", "o": "POS_X",
}
LTP_POS_MAP = {**POS_MAP, "nt": "POS_t"}
POS_TO_SHORT = {
    "POS_NOUN": "n", "POS_VERB": "v", "POS_ADJ": "a", "POS_ADV": "d",
    "POS_PRON": "r", "POS_DET": "det", "POS_ADP": "p", "POS_CCONJ": "cc",
    "POS_SCONJ": "sc", "POS_PART": "u", "POS_AUX": "aux", "POS_NUM": "m",
    "POS_PUNCT": "w", "POS_SYM": "sym", "POS_INTJ": "e", "POS_PROPN": "prop",
    "POS_X": "x", "POS_nr": "nr", "POS_ns": "ns", "POS_nt": "nt",
    "POS_nz": "nz", "POS_t": "t", "POS_f": "f",
}
SHORT_TO_POS = {
    "n": "POS_NOUN", "v": "POS_VERB", "a": "POS_ADJ", "d": "POS_ADV",
    "r": "POS_PRON", "det": "POS_DET", "p": "POS_ADP", "cc": "POS_CCONJ",
    "sc": "POS_SCONJ", "u": "POS_PART", "aux": "POS_AUX", "m": "POS_NUM",
    "w": "POS_PUNCT", "sym": "POS_SYM", "e": "POS_INTJ", "prop": "POS_PROPN",
    "x": "POS_X",
    "nr": "POS_nr", "ns": "POS_ns", "nt": "POS_nt", "nz": "POS_nz",
    "t": "POS_t", "f": "POS_f",
}


def atom_count(text):
    return sum(1 for _ in ATOM.finditer(text))


def normalize_text(text):
    return re.sub(r"[\t\r\f\v ]+", " ", text).strip()


def canonicalize_annotation(annotation):
    """Drop whitespace skipped by AtomCodec without merging adjacent words."""
    result = []
    for word, pos in annotation:
        word = re.sub(r"\s+", "", word)
        if word:
            result.append((word, pos))
    return result


def split_long(text, max_atoms):
    """Split conservatively without breaking an alphanumeric atom."""
    text = normalize_text(text)
    if not text:
        return
    for sentence in SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if atom_count(sentence) <= max_atoms:
            yield sentence
            continue
        parts = [part for part in SAFE_BREAK.split(sentence) if part]
        current = ""
        for part in parts:
            candidate = current + part
            if current and atom_count(candidate) > max_atoms:
                yield current.strip()
                current = part
            else:
                current = candidate
            while atom_count(current) > max_atoms:
                matches = list(ATOM.finditer(current))
                cut = matches[max_atoms].start()
                yield current[:cut].strip()
                current = current[cut:].strip()
        if current:
            yield current.strip()


def open_text(path):
    if path.suffix == ".zst":
        import zstandard
        raw = path.open("rb")
        return zstandard.ZstdDecompressor().stream_reader(raw)
    return path.open("rb")


def input_texts(paths, text_field):
    for path in paths:
        path = Path(path)
        import io
        with open_text(path) as raw, io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                data_suffix = Path(path.stem).suffix if path.suffix == ".zst" else path.suffix
                if data_suffix in (".jsonl", ".json"):
                    try:
                        value = json.loads(line)
                        text = value[text_field] if isinstance(value, dict) else value
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                else:
                    text = line
                if isinstance(text, str):
                    yield path, line_number, text


class PaddleTeacher:
    name = "paddlenlp-pos-tagging"

    def __init__(self, batch_size=64, device_id=0):
        import paddle
        from paddlenlp import Taskflow
        use_gpu = device_id >= 0 and paddle.is_compiled_with_cuda()
        paddle.set_device(f"gpu:{device_id}" if use_gpu else "cpu")
        self.model = Taskflow("pos_tagging", batch_size=batch_size)

    def annotate(self, texts):
        outputs = self.model(texts)
        if texts and outputs and isinstance(outputs[0], tuple):
            outputs = [outputs]
        return [canonicalize_annotation(
            [(word, POS_MAP.get(tag, "POS_X")) for word, tag in output]
        ) for output in outputs]


class LtpTeacher:
    def __init__(self, batch_size=64, device_id=0, model_id="LTP/small"):
        import torch
        from ltp import LTP
        self.name = f"ltp-cws-pos:{model_id}"
        self.batch_size = batch_size
        self.model = LTP(model_id)
        if device_id >= 0 and torch.cuda.is_available():
            self.model.to(f"cuda:{device_id}")

    def annotate(self, texts):
        output = self.model.pipeline(texts, tasks=["cws", "pos"])
        return [canonicalize_annotation(
            [(word, LTP_POS_MAP.get(tag, "POS_X")) for word, tag in zip(words, tags)]
        )
            for words, tags in zip(output.cws, output.pos)
        ]


def make_teacher(name, batch_size, device_id, model_id="LTP/small"):
    if name == "paddle":
        return PaddleTeacher(batch_size, device_id)
    if name == "ltp":
        return LtpTeacher(batch_size, device_id, model_id)
    raise ValueError(f"unknown external teacher: {name}")


class ShardWriter:
    def __init__(self, directory, prefix, shard_bytes, level=6):
        import zstandard
        self.zstandard = zstandard
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.level = level
        self.index = 0
        self.raw_bytes = 0
        self.lines = 0
        self.stream = None
        self.file = None
        self.files = []

    def _open(self):
        path = self.directory / f"{self.prefix}-{self.index:05d}.txt.zst"
        self.file = path.open("wb")
        self.stream = self.zstandard.ZstdCompressor(level=self.level).stream_writer(self.file)
        self.files.append(path)
        self.index += 1
        self.raw_bytes = 0

    def write(self, line):
        data = (line + "\n").encode()
        if self.stream is None or self.raw_bytes + len(data) > self.shard_bytes:
            self.close_stream()
            self._open()
        self.stream.write(data)
        self.raw_bytes += len(data)
        self.lines += 1

    def close_stream(self):
        if self.stream is not None:
            self.stream.flush(self.zstandard.FLUSH_FRAME)
            self.stream.close()
            self.file.close()
            self.stream = self.file = None

    def close(self):
        self.close_stream()


def validate_annotation(text, annotation, min_atoms, max_atoms):
    words = [word for word, _ in annotation]
    if not words or "".join(words).replace(" ", "") != text.replace(" ", ""):
        return "alignment"
    count = atom_count(text)
    if count < min_atoms:
        return "too_short"
    if count > max_atoms:
        return "too_long"
    if any(not word or "/" in word or "\n" in word for word in words):
        return "invalid_word"
    return None


def corpus_texts(path):
    """Yield reconstructed text from binary or word/POS gold corpora."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if fields:
                yield "".join(field.rpartition("/")[0] or field for field in fields)


def digest(text):
    return hashlib.blake2b(normalize_text(text).encode(), digest_size=16).digest()


def build(args):
    root = Path(args.output_root)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(cache_dir / f"dedup-{run_id}.sqlite3")
    database.execute("CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY, kind TEXT NOT NULL)")
    for leak_path in args.exclude_corpus:
        database.executemany(
            "INSERT OR IGNORE INTO seen VALUES (?, 'evaluation')",
            ((digest(text),) for text in corpus_texts(leak_path)),
        )
    database.commit()
    teacher = make_teacher(args.teacher, args.batch_size, args.device_id, args.model)
    prefix = f"train-{run_id}"
    binary = ShardWriter(root / "pseudo" / "binary", prefix, args.shard_mb * 1024**2)
    lac = ShardWriter(root / "pseudo" / "lac", prefix, args.shard_mb * 1024**2)
    rejected_writer = ShardWriter(root / "rejected", f"rejected-{run_id}", args.shard_mb * 1024**2)
    rejected = Counter()
    pos_counts = Counter()
    accepted = duplicate = source_lines = 0
    batch = []

    def consume(texts):
        nonlocal accepted, duplicate
        for text, annotation in zip(texts, teacher.annotate(texts)):
            reason = validate_annotation(text, annotation, args.min_atoms, args.max_atoms)
            if reason:
                rejected[reason] += 1
                rejected_writer.write(json.dumps({"reason": reason, "text": text}, ensure_ascii=False))
                continue
            text_digest = digest(text)
            try:
                database.execute("INSERT INTO seen VALUES (?, 'accepted')", (text_digest,))
            except sqlite3.IntegrityError:
                kind = database.execute("SELECT kind FROM seen WHERE digest = ?", (text_digest,)).fetchone()[0]
                rejected["evaluation_leak" if kind == "evaluation" else "duplicate"] += 1
                duplicate += kind != "evaluation"
                continue
            binary.write(" ".join(word for word, _ in annotation))
            lac.write(" ".join(f"{word}/{pos}" for word, pos in annotation))
            pos_counts.update(pos for _, pos in annotation)
            accepted += 1
        database.commit()

    started = time.time()
    try:
        for _path, _line, raw in input_texts(args.inputs, args.text_field):
            source_lines += 1
            for text in split_long(raw, args.max_atoms):
                batch.append(text)
                if len(batch) >= args.batch_size:
                    consume(batch)
                    batch.clear()
            if args.max_sentences and accepted >= args.max_sentences:
                break
        if batch:
            consume(batch)
    finally:
        binary.close()
        lac.close()
        rejected_writer.close()
        database.close()
    metadata = {
        "run_id": run_id, "teacher": teacher.name, "inputs": args.inputs,
        "exclude_corpus": args.exclude_corpus, "source_lines": source_lines,
        "accepted": accepted, "duplicates": duplicate, "rejected": dict(rejected),
        "min_atoms": args.min_atoms, "max_atoms": args.max_atoms,
        "binary_lines": binary.lines, "lac_lines": lac.lines,
        "pos_counts": dict(pos_counts.most_common()), "elapsed_seconds": time.time() - started,
        "binary_shards": [str(path) for path in binary.files],
        "lac_shards": [str(path) for path in lac.files],
        "rejected_shards": [str(path) for path in rejected_writer.files],
    }
    target = manifest_dir / f"pseudo-corpus-{run_id}.json"
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-root", default="/data/darts-corpus")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=108)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-mb", type=int, default=512)
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--teacher", choices=("paddle", "ltp"), default="paddle")
    parser.add_argument("--model", default="LTP/small", help="LTP model id; ignored by Paddle")
    parser.add_argument("--run-id", help="stable identifier used for resumable output names")
    parser.add_argument(
        "--exclude-corpus", action="append", default=[],
        help="gold binary/LAC corpus whose reconstructed sentences must never enter training",
    )
    args = parser.parse_args()
    if not 1 <= args.min_atoms <= args.max_atoms:
        parser.error("require 1 <= min-atoms <= max-atoms")
    build(args)


if __name__ == "__main__":
    main()
