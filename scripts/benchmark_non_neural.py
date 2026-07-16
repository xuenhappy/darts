#!/usr/bin/env python3
"""Validate and benchmark every runtime mode that does not depend on ONNX."""

import argparse
import json
from pathlib import Path
import statistics
import time

from darts import DSegment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODES = ("faster", "fast", "hybrid", "lac", "pinyin", "location")


def load_jsonc(path):
    return json.loads(
        "\n".join(
            line for line in Path(path).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("//")
        )
    )


def mode_depends_on_onnx(config, mode):
    mode_config = config["modes"][mode]
    plugins = [mode_config["decider"], *mode_config["recognizers"]]
    for section in ("deciders", "recognizers"):
        for name in plugins:
            plugin = config.get(section, {}).get(name)
            if plugin and plugin.get("type", "").startswith("Onnx"):
                return True
    return False


def load_sentences(path):
    sentences = []
    gold_boundaries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tokens = line.split()
            sentences.append("".join(tokens))
            offset = 0
            boundaries = set()
            for token in tokens[:-1]:
                offset += len(token)
                boundaries.add(offset)
            gold_boundaries.append(boundaries)
    return sentences, gold_boundaries


def signature(segment, text):
    atoms, output = segment.cut(text)
    atom_values = atoms.tolist()
    words = output.tolist()
    if not words:
        raise AssertionError(f"empty path for {text!r}")
    if words[0].atom_s != 0 or words[-1].atom_e != len(atom_values):
        raise AssertionError(f"incomplete path for {text!r}")
    if any(left.atom_e != right.atom_s for left, right in zip(words, words[1:])):
        raise AssertionError(f"discontinuous path for {text!r}")
    if "".join(word.image for word in words) != "".join(atom.image for atom in atom_values):
        raise AssertionError(f"surface mismatch for {text!r}")
    return tuple(
        (word.image, word.atom_s, word.atom_e, tuple(sorted(word.labels)))
        for word in words
    )


def validate(config_path, modes, sentences, repeat):
    segments = {mode: DSegment(config_path, mode) for mode in modes}
    signatures = {}
    for mode, segment in segments.items():
        outputs = []
        for text in sentences:
            expected = signature(segment, text)
            for _ in range(repeat - 1):
                if signature(segment, text) != expected:
                    raise AssertionError(f"{mode} is non-deterministic for {text!r}")
            outputs.append(expected)
        signatures[mode] = outputs

    if "hybrid" in signatures and "pinyin" in signatures:
        hybrid = [[item[:3] for item in row] for row in signatures["hybrid"]]
        pinyin = [[item[:3] for item in row] for row in signatures["pinyin"]]
        if hybrid != pinyin:
            raise AssertionError("pinyin mode changed hybrid segmentation boundaries")

    pinyin = segments.get("pinyin")
    if pinyin:
        result = signature(pinyin, "重庆音乐ABC")
        labels = {word: set(tags) for word, _start, _end, tags in result}
        if "chóng qìng" not in labels.get("重庆", set()):
            raise AssertionError("pinyin mode did not disambiguate 重庆")
        if "yīn yuè" not in labels.get("音乐", set()):
            raise AssertionError("pinyin mode did not disambiguate 音乐")

    lac = segments.get("lac")
    if lac:
        result = signature(lac, "中文分词在2026年发布")
        if not all(any(label.startswith("POS_") for label in tags) or
                   set(tags) & {"DATE", "DIGIT", "NUM", "POS", "SYMBOLS", "ENG", "CJK"}
                   for _word, _start, _end, tags in result):
            raise AssertionError("lac mode returned a token without POS/fallback labels")

    location = segments.get("location")
    if location:
        result = signature(location, "北京市海淀区中关村大街27号")
        labels = set().union(*(set(tags) for _word, _start, _end, tags in result))
        required = {"ADDR_PROVINCE", "ADDR_DISTRICT"}
        if not required <= labels:
            raise AssertionError(f"location mode missing labels: {required - labels}")

    return segments


def predicted_boundaries(segment, text):
    atoms, output = segment.cut(text)
    atom_values = atoms.tolist()
    return {
        atom_values[word.atom_e - 1].et
        for word in output.tolist()[:-1]
    }


def accuracy(segments, sentences, gold_boundaries):
    rows = {}
    predictions = {}
    for mode, segment in segments.items():
        mode_predictions = [
            predicted_boundaries(segment, text) for text in sentences
        ]
        predictions[mode] = mode_predictions
        true_positive = sum(
            len(predicted & gold)
            for predicted, gold in zip(mode_predictions, gold_boundaries)
        )
        predicted_total = sum(len(value) for value in mode_predictions)
        gold_total = sum(len(value) for value in gold_boundaries)
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[mode] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sentence_accuracy": sum(
                predicted == gold
                for predicted, gold in zip(mode_predictions, gold_boundaries)
            ) / len(sentences),
        }
    agreement = {}
    for left in segments:
        agreement[left] = {}
        for right in segments:
            agreement[left][right] = sum(
                a == b for a, b in zip(predictions[left], predictions[right])
            ) / len(sentences)
    return rows, agreement


def benchmark(segments, sentences, iterations, rounds, warmup):
    rows = []
    character_count = sum(len(text) for text in sentences)
    for mode, segment in segments.items():
        operation = lambda: [segment.cut(text) for text in sentences]
        for _ in range(warmup):
            operation()
        elapsed_rounds = []
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(iterations):
                operation()
            elapsed_rounds.append(time.perf_counter() - started)
        elapsed = statistics.median(elapsed_rounds)
        calls = iterations * len(sentences)
        rows.append({
            "mode": mode,
            "latency_ms": elapsed * 1000 / calls,
            "sentences_per_second": calls / elapsed,
            "characters_per_second": iterations * character_count / elapsed,
            "rounds": elapsed_rounds,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/conf.json")
    parser.add_argument("--corpus", default="data/demo/cws-dev.txt")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--determinism-repeat", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.rounds, args.determinism_repeat) < 1 or args.warmup < 0:
        parser.error("iterations, rounds and repeat must be positive; warmup must be non-negative")

    config = load_jsonc(args.config)
    modes = [
        mode for mode in DEFAULT_MODES
        if mode in config["modes"] and not mode_depends_on_onnx(config, mode)
    ]
    sentences, gold_boundaries = load_sentences(args.corpus)
    segments = validate(args.config, modes, sentences, args.determinism_repeat)
    accuracy_rows, agreement = accuracy(segments, sentences, gold_boundaries)
    rows = benchmark(segments, sentences, args.iterations, args.rounds, args.warmup)
    for row in rows:
        row.update(accuracy_rows[row["mode"]])
    if args.json:
        print(json.dumps({"modes": modes, "sentences": len(sentences),
                          "results": rows, "agreement": agreement},
                         ensure_ascii=False, indent=2))
        return
    print(f"validated={','.join(modes)} sentences={len(sentences)}")
    for row in rows:
        print(
            f"{row['mode']:10s} latency={row['latency_ms']:.4f}ms "
            f"throughput={row['sentences_per_second']:.1f} sent/s "
            f"chars={row['characters_per_second']:.0f}/s "
            f"F1={row['f1']:.4f} exact={row['sentence_accuracy']:.3f}"
        )
    print("exact-agreement:")
    print(" " * 11 + " ".join(f"{mode:>9s}" for mode in modes))
    for left in modes:
        print(f"{left:10s} " + " ".join(
            f"{agreement[left][right]:9.3f}" for right in modes
        ))


if __name__ == "__main__":
    main()
