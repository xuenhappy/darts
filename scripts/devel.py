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


def model_train(args):
    import torch
    from darts.devel.model import NerTrainer
    from darts.devel.reader import TorchNerSampleReader
    from darts.devel.sover import TSolver

    reader = TorchNerSampleReader(args.sample, args.labels, batch_nums=args.batch_size,
                                  max_words_len=args.max_words)
    model = NerTrainer(reader.wordsize(), args.hidden_size, reader.labelsize())
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if args.device == "cuda" and device != "cuda":
        raise RuntimeError("--device cuda requested but PyTorch cannot access CUDA")
    print(f"training device={device} samples={args.sample}")
    solver = TSolver(model, reader, {
        "model_outdir": args.output_dir,
        "epoch_num": args.epochs,
        "device": device,
        "log_every": args.log_every,
    })
    solver.solve()


def model_export(args):
    import torch

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = checkpoint.ner if hasattr(checkpoint, "ner") else checkpoint
    model.eval()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(output.parent)
        generated = Path(model.export2onnx())
        if generated.resolve() != output:
            generated.replace(output)
    finally:
        os.chdir(previous)
    print(f"exported={output}")


def build(args):
    command = ["bash", "scripts/build_all.sh"]
    if args.test:
        command.append("--test")
    subprocess.run(command, cwd=ROOT, check=True)


def data(args):
    command = [os.environ.get("PYTHON", "python3"), "scripts/data_pipeline.py", args.action]
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

    command = commands.add_parser("model-train", help="train the Transformer CRF recognizer")
    command.add_argument("sample")
    command.add_argument("--output-dir", default="model_bin")
    command.add_argument("--labels", default="O,B-_HWORD,I-_HWORD")
    command.add_argument("--epochs", type=int, default=6)
    command.add_argument("--batch-size", type=int, default=100)
    command.add_argument("--max-words", type=int, default=50)
    command.add_argument("--hidden-size", type=int, default=128)
    command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    command.add_argument("--log-every", type=int, default=50)
    command.set_defaults(func=model_train)

    command = commands.add_parser("model-export", help="export a trained Transformer checkpoint to ONNX")
    command.add_argument("checkpoint")
    command.add_argument("output")
    command.set_defaults(func=model_export)

    command = commands.add_parser("build", help="run the Meson-only build workflow")
    command.add_argument("--test", action="store_true")
    command.set_defaults(func=build)

    command = commands.add_parser("data", help="download, prepare, or build reproducible open data")
    command.add_argument("action", choices=("download", "prepare", "build-models"))
    command.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
