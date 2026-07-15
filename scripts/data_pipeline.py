#!/usr/bin/env python3
"""Reproducible open-data preparation and segmentation evaluation."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "sources.json"
DEFAULT_RAW = ROOT / "data" / "external"
DEFAULT_OUTPUT = ROOT / "data" / "generated"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir)
    lock_path = ROOT / "data" / "sources.lock.json"
    old_records = {}
    if lock_path.exists():
        old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        old_records = {record["name"]: record for record in old_lock.get("sources", [])}
    records = dict(old_records) if args.name else {}
    for source in manifest["sources"]:
        if args.name and source["name"] not in args.name:
            continue
        target = raw_dir / source["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        old_record = old_records.get(source["name"])
        if target.exists() and not args.force and old_record:
            actual = sha256(target)
            if actual != old_record.get("sha256"):
                raise IOError(f"checksum mismatch for {source['name']}; use --force to replace it")
        if args.force or not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "darts-data-pipeline/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(source["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0
                expected = response.headers.get("Content-Length")
                expected = offset + int(expected) if expected else None
                mode = "ab" if offset else "wb"
                with open(partial, mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if expected is not None and partial.stat().st_size != expected:
                raise IOError(f"incomplete download for {source['name']}: {partial.stat().st_size} != {expected}")
            partial.replace(target)
        record = dict(source)
        record["bytes"] = target.stat().st_size
        record["sha256"] = sha256(target)
        records[source["name"]] = record
        print(f"ready {source['name']}: {target} ({record['bytes']} bytes)")
    ordered = [records[source["name"]] for source in manifest["sources"] if source["name"] in records]
    lock_path.write_text(json.dumps({"sources": ordered}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"lock={lock_path}")


def read_conllu(path):
    sentence = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) != 10 or "-" in columns[0] or "." in columns[0]:
                continue
            sentence.append((columns[1], columns[3]))
    if sentence:
        yield sentence


def write_segmented(source, target):
    count = 0
    with open(target, "w", encoding="utf-8") as output:
        for sentence in read_conllu(source):
            output.write(" ".join(word for word, _pos in sentence) + "\n")
            count += 1
    return count


def prepare(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    train_sentences = list(read_conllu(raw_dir / "ud-chinese-gsd" / "train.conllu"))
    for split in ("train", "dev", "test"):
        source = raw_dir / "ud-chinese-gsd" / f"{split}.conllu"
        target = output_dir / f"cws-{split}.txt"
        splits[split] = write_segmented(source, target)

    words = Counter()
    word_pos = {}
    ud_words = set()
    singles = Counter()
    bigrams = Counter()
    for sentence in train_sentences:
        forms = [word for word, _pos in sentence]
        for word, pos in sentence:
            words[word] += 1
            ud_words.add(word)
            word_pos.setdefault(word, pos)
            singles[word] += 1
        bigrams.update(zip(forms, forms[1:]))

    jieba_path = raw_dir / "jieba" / "dict.txt"
    with open(jieba_path, encoding="utf-8") as stream:
        for line in stream:
            columns = line.rstrip().split()
            if not columns:
                continue
            word = columns[0]
            frequency = int(columns[1]) if len(columns) > 1 and columns[1].isdigit() else 1
            pos = columns[2] if len(columns) > 2 else "X"
            words[word] = max(words[word], frequency)
            word_pos.setdefault(word, f"JIEBA_{pos}")

    dictionary_path = output_dir / "dictionary.txt"
    selected_words = set()
    with open(dictionary_path, "w", encoding="utf-8") as output:
        for word in sorted(words):
            if len(word) < 2 or len(word) > args.max_word_length:
                continue
            if word not in ud_words and words[word] < args.jieba_min_frequency:
                continue
            label = word_pos[word].replace("<", "").replace(">", "")
            output.write(f"<CJK>,<{label}>:{word}\n")
            selected_words.add(word)

    phrase_path = output_dir / "pinyin-phrases.txt"
    phrase_entries = 0
    with open(raw_dir / "phrase-pinyin-data" / "pinyin.txt", encoding="utf-8") as source, \
            open(phrase_path, "w", encoding="utf-8") as output:
        output.write("# generated from phrase-pinyin-data v0.19.0; MIT\n")
        for line in source:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            phrase, readings = (part.strip() for part in line.split(":", 1))
            syllables = readings.split()
            if phrase in selected_words and len(phrase) == len(syllables):
                output.write(f"{phrase}: {' '.join(syllables)}\n")
                phrase_entries += 1

    single_path = output_dir / "bigram-single.txt"
    with open(single_path, "w", encoding="utf-8") as output:
        for word, frequency in sorted(singles.items()):
            if "-" not in word:
                output.write(f"{word}\t{frequency}\n")
    union_path = output_dir / "bigram-union.txt"
    with open(union_path, "w", encoding="utf-8") as output:
        for (left, right), frequency in sorted(bigrams.items()):
            if "-" not in left and "-" not in right:
                output.write(f"{left}-{right}\t{frequency}\n")

    metadata = {
        "splits": splits,
        "dictionary_entries": sum(1 for _ in open(dictionary_path, encoding="utf-8")),
        "single_entries": len(singles),
        "bigram_entries": len(bigrams),
        "jieba_min_frequency": args.jieba_min_frequency,
        "max_word_length": args.max_word_length,
        "pinyin_phrase_entries": phrase_entries,
        "licenses": ["UD Chinese GSD: CC BY-SA 4.0", "jieba dictionary: MIT",
                     "phrase-pinyin-data: MIT"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def build_models(args):
    from darts import build_gramdict_fromfile, compileDregex

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    def dictionary_entries():
        with open(output_dir / "dictionary.txt", encoding="utf-8") as stream:
            for line in stream:
                labels, word = line.rstrip("\n").split(":", 1)
                yield word, labels.replace("<", "").replace(">", "")

    dictionary = model_dir / "open-dictionary.pbs"
    bigram = model_dir / "open-bigram.bdf"
    compileDregex(dictionary_entries(), str(dictionary))
    build_gramdict_fromfile(str(output_dir / "bigram-single.txt"),
                            str(output_dir / "bigram-union.txt"), str(bigram))
    config = {
        "dservices": {},
        "recognizers": {
            "open.dict": {"type": "DictWordRecongnizer", "pbfile.path": str(dictionary),
                          "atom.mode": "false"}
        },
        "deciders": {
            "open.min": {"type": "MinCoverDecider"},
            "open.bigram": {"type": "BigramDecider", "dat.path": str(bigram)},
            "open.hybrid": {"type": "HybridStatDecider", "dat.path": str(bigram),
                            "bigram.weight": "2.0", "length.weight": "15.0",
                            "length.power": "1.0", "token.penalty": "0.0",
                            "unknown.penalty": "20.0"},
        },
        "modes": {
            "open-faster": {"decider": "open.min", "recognizers": ["open.dict"]},
            "open-fast": {"decider": "open.bigram", "recognizers": ["open.dict"]},
            "open-hybrid": {"decider": "open.hybrid", "recognizers": ["open.dict"]},
        },
        "default.mode": "open-hybrid",
    }
    config_path = output_dir / "open-conf.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"dictionary={dictionary}\nbigram={bigram}\nconfig={config_path}")


def atom_count(text):
    from darts import PyAtomList
    return len(PyAtomList(text, skip_space=True, normal_before=False))


def evaluate(args):
    from darts import DSegment

    segment = DSegment(args.config, args.mode)
    true_positive = predicted_total = gold_total = exact = sentences = 0
    with open(args.gold, encoding="utf-8") as stream:
        for line in stream:
            tokens = line.strip().split()
            if not tokens:
                continue
            text = "".join(tokens)
            gold_boundaries = set()
            position = 0
            for token in tokens[:-1]:
                position += atom_count(token)
                gold_boundaries.add(position)
            atoms, output = segment.cut(text)
            predicted_boundaries = {word.atom_e for word in output.tolist()[:-1]}
            true_positive += len(gold_boundaries & predicted_boundaries)
            predicted_total += len(predicted_boundaries)
            gold_total += len(gold_boundaries)
            exact += predicted_boundaries == gold_boundaries
            sentences += 1
            if args.limit and sentences >= args.limit:
                break
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(json.dumps({"sentences": sentences, "precision": precision, "recall": recall,
                      "f1": f1, "sentence_exact": exact / sentences if sentences else 0.0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download")
    command.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    command.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    command.add_argument("--force", action="store_true")
    command.add_argument("--name", action="append", help="download only the named manifest entry")
    command.set_defaults(func=download)
    command = commands.add_parser("prepare")
    command.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    command.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    command.add_argument("--jieba-min-frequency", type=int, default=10)
    command.add_argument("--max-word-length", type=int, default=8)
    command.set_defaults(func=prepare)
    command = commands.add_parser("evaluate")
    command.add_argument("gold")
    command.add_argument("--config", default=str(ROOT / "data" / "conf.json"))
    command.add_argument("--mode", default="faster")
    command.add_argument("--limit", type=int, default=0)
    command.set_defaults(func=evaluate)
    command = commands.add_parser("build-models")
    command.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    command.set_defaults(func=build_models)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
