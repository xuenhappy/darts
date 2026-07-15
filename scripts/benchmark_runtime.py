#!/usr/bin/env python3
"""Measure atomization and segmentation throughput using the installed package."""

import argparse
import time

from darts import DSegment, PyAtomList


def measure(name, operation, iterations, warmup):
    for _ in range(warmup):
        operation()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    elapsed = time.perf_counter() - started
    print(
        f"{name:12s} total={elapsed:.6f}s "
        f"latency={elapsed * 1000 / iterations:.3f}ms "
        f"throughput={iterations / elapsed:.1f} calls/s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/conf.json")
    parser.add_argument("--text", default="目标检测模型量化和中文分词性能测试")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")

    measure("atomize", lambda: PyAtomList(args.text), args.iterations, args.warmup)
    for mode in ("faster", "fast"):
        segment = DSegment(args.config, mode)
        measure(mode, lambda segment=segment: segment.cut(args.text), args.iterations, args.warmup)


if __name__ == "__main__":
    main()
