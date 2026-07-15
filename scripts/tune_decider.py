#!/usr/bin/env python3
"""Grid-search HybridStatDecider parameters on a segmented development set."""

import argparse
import itertools
import json
from pathlib import Path
import tempfile

from darts import DSegment, PyAtomList


ROOT = Path(__file__).resolve().parents[1]


def atom_count(text):
    return len(PyAtomList(text, skip_space=True, normal_before=False))


def evaluate(config, gold):
    segment = DSegment(str(config), "hybrid")
    true_positive = predicted_total = gold_total = exact = sentences = 0
    with open(gold, encoding="utf-8") as stream:
        for line in stream:
            tokens = line.strip().split()
            if not tokens:
                continue
            position = 0
            gold_boundaries = set()
            for token in tokens[:-1]:
                position += atom_count(token)
                gold_boundaries.add(position)
            _atoms, output = segment.cut("".join(tokens))
            predicted = {word.atom_e for word in output.tolist()[:-1]}
            true_positive += len(gold_boundaries & predicted)
            predicted_total += len(predicted)
            gold_total += len(gold_boundaries)
            exact += predicted == gold_boundaries
            sentences += 1
    precision = true_positive / predicted_total
    recall = true_positive / gold_total
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "sentence_exact": exact / sentences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(ROOT / "data/generated/cws-dev.txt"))
    parser.add_argument("--dictionary", default=str(ROOT / "data/models/panda.pbs"))
    parser.add_argument("--bigram", default=str(ROOT / "data/models/ngram_dict.bdf"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--fine", action="store_true", help="search around the best coarse-grid region")
    args = parser.parse_args()
    if args.fine:
        grid = itertools.product((2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
                                 (10.0, 15.0, 20.0, 25.0, 30.0),
                                 (6.0, 8.0, 10.0, 12.0, 15.0, 20.0), (0.0,))
    else:
        grid = itertools.product((0.25, 0.5, 1.0, 2.0, 4.0), (20.0, 50.0, 100.0),
                                 (0.0, 2.0, 5.0, 10.0), (0.0, 1.0, 2.0))
    results = []
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "config.json"
        for bigram_weight, length_weight, unknown_penalty, token_penalty in grid:
            config = {
                "dservices": {},
                "recognizers": {"dict": {"type": "DictWordRecongnizer",
                                             "pbfile.path": str(Path(args.dictionary).resolve()),
                                             "atom.mode": "false"}},
                "deciders": {"hybrid": {"type": "HybridStatDecider",
                                            "dat.path": str(Path(args.bigram).resolve()),
                                            "bigram.weight": str(bigram_weight),
                                            "length.weight": str(length_weight),
                                            "length.power": "1.0",
                                            "unknown.penalty": str(unknown_penalty),
                                            "token.penalty": str(token_penalty)}},
                "modes": {"hybrid": {"decider": "hybrid", "recognizers": ["dict"]}},
                "default.mode": "hybrid",
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            metrics = evaluate(config_path, args.gold)
            metrics["params"] = {"bigram.weight": bigram_weight, "length.weight": length_weight,
                                  "unknown.penalty": unknown_penalty, "token.penalty": token_penalty}
            results.append(metrics)
    results.sort(key=lambda item: (item["f1"], item["sentence_exact"]), reverse=True)
    print(json.dumps(results[:args.top], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
