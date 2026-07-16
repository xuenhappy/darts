#!/usr/bin/env python3
"""Development CLI for dictionary compilation, benchmarking, and model training."""

import argparse
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def read_dictionary(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{line_number}: expected LABELS:WORD")
            labels, word = line.split(":", 1)
            labels = labels.replace("<", "").replace(">", "")
            if word:
                yield word, labels


def dict_compile(args):
    from darts import compileDregex

    started = time.perf_counter()
    compileDregex(read_dictionary(args.input), args.output)
    elapsed = time.perf_counter() - started
    size = Path(args.output).stat().st_size
    print(f"compiled={args.output} size={size} bytes elapsed={elapsed:.3f}s")


def dict_repack(args):
    executable = ROOT / args.build_dir / "darts-dict-repack"
    if not executable.exists():
        raise FileNotFoundError(f"build the repack tool first: {executable}")
    subprocess.run([str(executable), args.input, args.output], check=True)
    old_size = Path(args.input).stat().st_size
    new_size = Path(args.output).stat().st_size
    print(f"old={old_size} new={new_size} ratio={new_size / old_size:.4f}")


def dict_benchmark(args):
    from darts import Dregex, PyAtomList

    started = time.perf_counter()
    regex = Dregex(args.dictionary)
    load_elapsed = time.perf_counter() - started
    atoms = [atom.image for atom in PyAtomList(args.text).tolist()]
    hits = 0

    def hit(_start, _end, _labels):
        nonlocal hits
        hits += 1
        return False

    for _ in range(args.warmup):
        regex.parse(atoms, hit)
    hits = 0
    started = time.perf_counter()
    for _ in range(args.repeat):
        regex.parse(atoms, hit)
    elapsed = time.perf_counter() - started
    print(
        f"load={load_elapsed:.3f}s latency={elapsed * 1000 / args.repeat:.3f}ms "
        f"throughput={args.repeat / elapsed:.1f} calls/s hits={hits}"
    )


def bigram_compile(args):
    from darts import build_gramdict_fromfile

    started = time.perf_counter()
    build_gramdict_fromfile(args.single, args.bigram, args.output)
    elapsed = time.perf_counter() - started
    size = Path(args.output).stat().st_size
    print(f"compiled={args.output} size={size} bytes elapsed={elapsed:.3f}s")


def model_train(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_recognizer.py", "train",
               "--train", args.sample, "--dev", args.dev, "--epochs", str(args.epochs),
               "--output-dir", args.output_dir, "--batch-size", str(args.batch_size),
               "--max-span", str(args.max_span), "--hidden-size", str(args.hidden_size),
               "--device", args.device]
    subprocess.run(command, cwd=ROOT, check=True)


def model_export(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_recognizer.py", "export",
               args.checkpoint, args.output]
    subprocess.run(command, cwd=ROOT, check=True)


def syntax_train(args):
    command = [
        os.environ.get("PYTHON", "python3"), "scripts/train_syntax_recognizer.py", "train",
        "--train", args.train, "--dev", args.dev, "--type-map", args.type_map,
        "--output-dir", args.output_dir, "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--max-span", str(args.max_span),
        "--hidden-size", str(args.hidden_size), "--device", args.device,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def syntax_export(args):
    command = [
        os.environ.get("PYTHON", "python3"), "scripts/train_syntax_recognizer.py", "export",
        args.checkpoint, args.output,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def quantizer_train(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_quantizer.py", "train",
               "--train", args.train, "--dev", args.dev, "--epochs", str(args.epochs),
               "--output-dir", args.output_dir, "--batch-size", str(args.batch_size),
               "--hidden-size", str(args.hidden_size), "--device", args.device]
    subprocess.run(command, cwd=ROOT, check=True)


def quantizer_export(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_quantizer.py", "export",
               args.checkpoint, args.output_dir]
    subprocess.run(command, cwd=ROOT, check=True)


def joint_train(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_joint.py", "train",
               "--epochs", str(args.epochs),
               "--output-dir", args.output_dir, "--max-span", str(args.max_span),
               "--hidden-size", str(args.hidden_size), "--device", args.device,
               "--recognizer-kind", args.recognizer_kind]
    if args.train:
        command.extend(["--train", args.train])
    if args.dev:
        command.extend(["--dev", args.dev])
    if args.mode:
        command.extend(["--mode", args.mode])
    if args.type_map:
        command.extend(["--type-map", args.type_map])
    if args.resume:
        command.extend(["--resume", args.resume])
    if not args.amp:
        command.append("--no-amp")
    subprocess.run(command, cwd=ROOT, check=True)


def joint_export(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_joint.py", "export",
               args.checkpoint, args.output_dir]
    subprocess.run(command, cwd=ROOT, check=True)


def joint_evaluate(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/train_joint.py", "evaluate",
               args.checkpoint, "--data", args.data, "--device", args.device]
    if args.mode:
        command.extend(["--mode", args.mode])
    if args.type_map:
        command.extend(["--type-map", args.type_map])
    subprocess.run(command, cwd=ROOT, check=True)


def build(args):
    command = ["bash", "scripts/build_all.sh"]
    if args.test:
        command.append("--test")
    subprocess.run(command, cwd=ROOT, check=True)


def data(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/data_pipeline.py", args.action]
    subprocess.run(command, cwd=ROOT, check=True)


def tmap_build(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/build_tmap.py",
               "--manual", args.manual, "--output", args.output,
               "--min-frequency", str(args.min_frequency),
               "--max-new-characters", str(args.max_new_characters)]
    for path in args.finefreq:
        command.extend(["--finefreq", path])
    subprocess.run(command, cwd=ROOT, check=True)


def location_build(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/location_data.py",
               "--source", args.source, "--poi", args.poi,
               "--output-dir", args.output_dir, "--work-dir", args.work_dir]
    subprocess.run(command, cwd=ROOT, check=True)


def location_poi_extract(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/extract_osm_poi.py",
               args.input, "--output", args.output]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("dict-compile", help="compile LABELS:WORD text into a .pbs dictionary")
    command.add_argument("input")
    command.add_argument("output")
    command.set_defaults(func=dict_compile)

    command = commands.add_parser("dict-repack", help="convert a legacy .pbs dictionary to compact v2")
    command.add_argument("input")
    command.add_argument("output")
    command.add_argument("--build-dir", default="build/meson")
    command.set_defaults(func=dict_repack)

    command = commands.add_parser("dict-benchmark", help="measure dictionary load and parse throughput")
    command.add_argument("dictionary")
    command.add_argument("--text", default="龟鹿二仙胶和龟鹤延年汤")
    command.add_argument("--repeat", type=int, default=10000)
    command.add_argument("--warmup", type=int, default=100)
    command.set_defaults(func=dict_benchmark)

    command = commands.add_parser(
        "bigram-compile", help="compile unigram/bigram counts into a smoothed .bdf model"
    )
    command.add_argument("single", help="tab-separated WORD/FREQUENCY file")
    command.add_argument("bigram", help="tab-separated LEFT-RIGHT/FREQUENCY file")
    command.add_argument("output")
    command.set_defaults(func=bigram_compile)

    command = commands.add_parser("model-train", help="train the overlapping-span Transformer recognizer")
    command.add_argument("sample")
    command.add_argument("--output-dir", default="model_bin")
    command.add_argument("--dev", default="data/generated/cws-dev.txt")
    command.add_argument("--epochs", type=int, default=6)
    command.add_argument("--batch-size", type=int, default=100)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=model_train)

    command = commands.add_parser("model-export", help="export a trained Transformer checkpoint to ONNX")
    command.add_argument("checkpoint")
    command.add_argument("output")
    command.set_defaults(func=model_export)

    command = commands.add_parser(
        "syntax-train", help="train a NOT_WORD plus POS multi-class span recognizer"
    )
    command.add_argument("--train", default="data/generated/lac-train.txt")
    command.add_argument("--dev", default="data/generated/lac-dev.txt")
    command.add_argument("--type-map", default="data/codes/pos.hx.txt")
    command.add_argument("--output-dir", default="model_bin/syntax")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=syntax_train)

    command = commands.add_parser(
        "syntax-export", help="export the multi-class syntax span recognizer to ONNX"
    )
    command.add_argument("checkpoint")
    command.add_argument("output")
    command.set_defaults(func=syntax_export)

    command = commands.add_parser("quantizer-train", help="train the Transformer graph quantizer")
    command.add_argument("--train")
    command.add_argument("--dev")
    command.add_argument("--output-dir", default="model_bin/quantizer")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.set_defaults(func=quantizer_train)

    command = commands.add_parser("quantizer-export", help="export indicator and quantizer ONNX models")
    command.add_argument("checkpoint")
    command.add_argument("output_dir")
    command.set_defaults(func=quantizer_export)

    command = commands.add_parser("joint-train", help="jointly train one encoder for both neural tasks")
    command.add_argument("--train", default="data/generated/cws-train.txt")
    command.add_argument("--dev", default="data/generated/cws-dev.txt")
    command.add_argument("--output-dir", default="model_bin/joint")
    command.add_argument("--epochs", type=int, default=20)
    command.add_argument("--max-span", type=int, default=5)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.add_argument("--recognizer-kind", choices=("binary", "syntax"), default="binary")
    command.add_argument("--mode")
    command.add_argument("--type-map")
    command.add_argument("--resume")
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    command.set_defaults(func=joint_train)

    command = commands.add_parser("joint-export", help="export both tasks from a joint checkpoint")
    command.add_argument("checkpoint")
    command.add_argument("output_dir")
    command.set_defaults(func=joint_export)

    command = commands.add_parser("joint-evaluate", help="evaluate a joint checkpoint on held-out data")
    command.add_argument("checkpoint")
    command.add_argument("--data", default="data/generated/cws-test.txt")
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.add_argument("--mode")
    command.add_argument("--type-map")
    command.set_defaults(func=joint_evaluate)

    command = commands.add_parser("build", help="run the Meson-only build workflow")
    command.add_argument("--test", action="store_true")
    command.set_defaults(func=build)

    command = commands.add_parser("data", help="download, prepare, or build reproducible open data")
    command.add_argument("action", choices=("download", "prepare", "build-models"))
    command.set_defaults(func=data)

    command = commands.add_parser(
        "tmap-build", help="build chars.tmap from curated and FineFreq character data"
    )
    command.add_argument("--manual", default="data/kernel/chars.manual.tmap")
    command.add_argument("--finefreq", action="append", default=[])
    command.add_argument("--output", default="data/kernel/chars.tmap")
    command.add_argument("--min-frequency", type=int, default=100_000)
    command.add_argument("--max-new-characters", type=int, default=4096)
    command.set_defaults(func=tmap_build)

    command = commands.add_parser("location-build", help="build address dictionary and quantizer data")
    command.add_argument("--source", default="data/external/province-city-china/data.csv")
    command.add_argument("--poi", default="data/external/location/poi.txt")
    command.add_argument("--output-dir", default="data/models")
    command.add_argument("--work-dir", default="data/generated/location")
    command.set_defaults(func=location_build)

    command = commands.add_parser("location-poi-extract", help="extract POI names from an OSM PBF")
    command.add_argument("input")
    command.add_argument("--output", default="data/external/location/poi.txt")
    command.set_defaults(func=location_poi_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
