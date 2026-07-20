#!/usr/bin/env python3
"""Darts end-to-end demonstration with difficult mixed-language inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Iterable

from darts import LocationSegmenter, PinyinAnnotator, Tokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data" / "conf.json"

COMPLEX_TEXTS = (
    "南京市长江大桥研究院计划于2026年7月20日14:30发布Darts-v2.0.1，"
    "模型占用2.5GB显存，推理速度达到12.8万字/秒。",
    "李明在中国科学院计算技术研究所负责自然语言处理、GraphLossSparse和"
    "Transformer量化器优化，OpenAI-compatible API的P99延迟下降了37.5%。",
    "银行行长在重庆音乐厅说：“他说的确实在理”，随后用招商银行App转账￥12,580.60。",
    "用户访问https://github.com/xuenhappy/darts，提交issue #128并发送邮件到"
    "nlp-team@example.com；版本号为v2.0.1-rc2。",
    "小明硕士毕业于中国科学院计算所，后在日本京都大学深造，研究生命起源与"
    "机器学习；乒乓球拍卖完了并不等于乒乓球拍卖。",
)

AMBIGUOUS_TEXTS = (
    "南京市长江大桥",
    "武汉市长江大桥",
    "研究生命起源",
    "他说的确实在理",
    "结婚的和尚未结婚的",
    "乒乓球拍卖完了",
)

ADDRESS_TEXTS = (
    "浙江省杭州市西湖区文三路90号东部软件园2号楼",
    "北京市海淀区中关村街道科学院南路88号融科资讯中心A座",
    "上海市浦东新区世纪大道100号环球金融中心28层",
)


def title(text: str) -> None:
    print(f"\n{'=' * 18} {text} {'=' * 18}")


def render_tokens(tokenizer: Tokenizer, text: str, show_labels: bool = True) -> None:
    tokens = tokenizer.tokens(text)
    print(f"\n原文: {text}")
    print("切分:", " | ".join(token.text for token in tokens))
    print("偏移:")
    for token in tokens:
        labels = ",".join(sorted(token.labels)) if show_labels else "-"
        print(f"  [{token.start:>3}, {token.end:<3}) {token.text:<20} {labels}")


def run_segmentation(config: Path, mode: str, texts: Iterable[str]) -> None:
    title(f"{mode or 'default'} 最佳路径")
    tokenizer = Tokenizer(str(config), mode)
    for text in texts:
        render_tokens(tokenizer, text)


def run_lac(config: Path, mode: str, text: str) -> None:
    title(f"{mode} 分词与词性")
    tokenizer = Tokenizer(str(config), mode)
    tokens = tokenizer.lac(text)
    print("原文:", text)
    print("LAC :", " | ".join(f"{token.text}/{token.pos}" for token in tokens))
    for token in tokens:
        print(
            f"  [{token.start:>3}, {token.end:<3}) "
            f"{token.text:<20} POS={token.pos:<8} labels={','.join(sorted(token.labels))}"
        )


def run_search(config: Path, text: str, limit: int) -> None:
    title("重叠候选词")
    tokenizer = Tokenizer(str(config), "hybrid")
    candidates = list(tokenizer.tokenize(text, mode="search"))
    candidates.sort(key=lambda item: (item[1], -(item[2] - item[1]), item[0]))
    print("原文:", text)
    print(f"候选数: {len(candidates)}，显示前 {min(limit, len(candidates))} 个")
    for word, start, end in candidates[:limit]:
        print(f"  [{start:>2}, {end:<2}) {word}")


def run_sampling(config: Path, temperature: float, samples: int) -> None:
    title(f"歧义路径采样 T={temperature:g}")
    tokenizer = Tokenizer(str(config), "hybrid")
    for text in AMBIGUOUS_TEXTS:
        deterministic = tokenizer.lcut(text)
        alternatives = {
            tuple(tokenizer.sample(text, temperature=temperature)) for _ in range(samples)
        }
        print(f"\n原文: {text}")
        print("T=0 :", " | ".join(deterministic))
        for index, alternative in enumerate(sorted(alternatives), 1):
            print(f"T>0 #{index:<2}:", " | ".join(alternative))


def run_pinyin(config: Path) -> None:
    title("句子级拼音消歧")
    annotator = PinyinAnnotator(str(config))
    text = "重庆银行行长带着音乐学院学生前往长安大街，购买了3台ThinkPad。"
    print("原文:", text)
    for token in annotator.annotate(text):
        reading = token.pinyin if token.pinyin is not None else "<非中文>"
        print(f"  [{token.start:>2}, {token.end:<2}) {token.text:<12} {reading}")
    print("拼音:", annotator.format(text, preserve_non_cjk=True))


def run_locations(config: Path) -> None:
    title("地址与 POI 结构化解析")
    segmenter = LocationSegmenter(str(config))
    for address in ADDRESS_TEXTS:
        result = segmenter.parse(address)
        print(f"\n地址: {address}")
        print("切分:", " | ".join(f"{token.text}/{token.kind}" for token in result.tokens))
        print("组件:", result.components)


def run_benchmark(config: Path, mode: str, rounds: int) -> None:
    title(f"{mode} 简单吞吐测试")
    tokenizer = Tokenizer(str(config), mode)
    texts = COMPLEX_TEXTS * 20
    total_chars = sum(map(len, texts)) * rounds

    # Warm up lazy allocations and ONNX kernels before timing steady-state work.
    for text in texts[:5]:
        tokenizer.lcut(text)
    started = time.perf_counter()
    for _ in range(rounds):
        tokenizer.batch_cut(texts)
    elapsed = time.perf_counter() - started
    print(
        f"{total_chars:,} 字符 / {elapsed:.3f} 秒 = "
        f"{total_chars / elapsed:,.0f} 字符/秒"
    )


def optional(name: str, operation) -> None:
    try:
        operation()
    except (OSError, RuntimeError) as error:
        print(f"\n[跳过 {name}] 当前安装或配置未提供所需模型: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", default="hybrid", help="主演示及测速模式")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--no-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    if not config.is_file():
        raise SystemExit(f"配置文件不存在: {config}")

    run_segmentation(config, args.mode, COMPLEX_TEXTS)
    run_search(config, COMPLEX_TEXTS[0], args.candidate_limit)
    run_sampling(config, args.temperature, args.samples)
    run_pinyin(config)
    run_locations(config)
    optional("lac", lambda: run_lac(config, "lac", COMPLEX_TEXTS[0]))
    optional("neural", lambda: run_segmentation(config, "neural", COMPLEX_TEXTS[:2]))
    optional("lac-nural", lambda: run_lac(config, "lac-nural", COMPLEX_TEXTS[0]))
    if not args.no_benchmark:
        run_benchmark(config, args.mode, args.rounds)


if __name__ == "__main__":
    main()
