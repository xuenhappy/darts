#!/usr/bin/env python3
"""Stream clean article paragraphs from a Wikimedia XML dump to JSONL.zst."""

import argparse
import bz2
from collections import Counter
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import mwparserfromhell
import zstandard


SPACE = re.compile(r"[\t\r\f\v ]+")
BLANKS = re.compile(r"\n{2,}")
NOISE = re.compile(r"^(?:category|file|image|template|分类|文件|图像):", re.IGNORECASE)


def clean_wikitext(text):
    """Remove wiki markup while retaining visible link and heading text."""
    if not text:
        return []
    code = mwparserfromhell.parse(text)
    plain = code.strip_code(normalize=True, collapse=True)
    paragraphs = []
    for paragraph in BLANKS.split(plain):
        paragraph = SPACE.sub(" ", paragraph.replace("\n", " ")).strip()
        if paragraph and not NOISE.match(paragraph):
            paragraphs.append(paragraph)
    return paragraphs


def local_name(tag):
    return tag.rpartition("}")[2]


def pages(path):
    with bz2.open(path, "rb") as source:
        for _event, element in ET.iterparse(source, events=("end",)):
            if local_name(element.tag) != "page":
                continue
            title = namespace = text = None
            redirect = False
            for child in element.iter():
                name = local_name(child.tag)
                if name == "title" and title is None:
                    title = child.text or ""
                elif name == "ns" and namespace is None:
                    namespace = child.text
                elif name == "redirect":
                    redirect = True
                elif name == "text":
                    text = child.text or ""
            yield title or "", namespace, redirect, text or ""
            element.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--opencc", default="t2s", choices=("none", "t2s", "s2t"))
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-paragraph-chars", type=int, default=20000)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()
    converter = None
    if args.opencc != "none":
        from opencc import OpenCC
        converter = OpenCC(args.opencc)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    started = time.time()
    compressor = zstandard.ZstdCompressor(level=6, threads=-1)
    with target.open("wb") as raw, compressor.stream_writer(raw) as output:
        for title, namespace, redirect, text in pages(args.input):
            counts["pages"] += 1
            if namespace != "0":
                counts["non_article"] += 1
                continue
            if redirect:
                counts["redirect"] += 1
                continue
            if converter:
                title = converter.convert(title)
            for paragraph_index, paragraph in enumerate(clean_wikitext(text)):
                if converter:
                    paragraph = converter.convert(paragraph)
                if len(paragraph) < args.min_chars:
                    counts["too_short"] += 1
                    continue
                if len(paragraph) > args.max_paragraph_chars:
                    counts["too_long"] += 1
                    continue
                record = {"text": paragraph, "title": title, "paragraph": paragraph_index,
                          "source": "zhwiki"}
                output.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                counts["paragraphs"] += 1
                counts["characters"] += len(paragraph)
            counts["accepted_pages"] += 1
            if args.max_pages and counts["accepted_pages"] >= args.max_pages:
                break
    manifest = {
        "input": args.input, "output": args.output, "opencc": args.opencc,
        "min_chars": args.min_chars, "max_paragraph_chars": args.max_paragraph_chars,
        "counts": dict(counts), "elapsed_seconds": time.time() - started,
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
