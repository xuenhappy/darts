#!/usr/bin/env python3
"""Update the runtime character-type map with optional frequency data."""

import argparse
import csv
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TMAP = ROOT / "data" / "kernel" / "chars.tmap"
TYPE_ORDER = ("SYMBOLS", "POS", "ARROW", "FACE", "RUSH", "ORDER",
              "MOTH", "DAY", "TIME", "ENG", "NUM")
PUNCTUATION_CATEGORIES = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}
SYMBOL_CATEGORIES = {"Sc", "Sk", "Sm", "So"}


def read_tmap(path):
    """Read a tmap while preserving the last-assignment-wins runtime rule."""
    mapping = {}
    current_type = None
    with open(path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("-%"):
                current_type = line[2:].strip()
                if not current_type:
                    raise ValueError(f"{path}:{line_number}: empty type name")
                continue
            if current_type is None:
                raise ValueError(f"{path}:{line_number}: characters before first type")
            for character in line:
                mapping[character] = current_type
    return mapping


def is_cjk(character):
    codepoint = ord(character)
    return (
        codepoint in (0x3005, 0x3007, 0x303B)
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2EE5F
        or 0x2F800 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x3347F
    )


def inferred_type(character, category, unicode_name):
    """Return a safe Atom type for common Chinese/English web characters."""
    if is_cjk(character) or category.startswith("Z") or category.startswith("C"):
        return None
    if category == "Nd":
        return "NUM"
    if category in PUNCTUATION_CATEGORIES:
        return "POS"
    if category in {"Ll", "Lm", "Lt", "Lu", "Mc", "Me", "Mn"} and "LATIN" in unicode_name:
        return "ENG"
    if category in SYMBOL_CATEGORIES:
        return "SYMBOLS"
    return None


def add_finefreq(mapping, paths, min_frequency, max_new_characters):
    candidates = {}
    for path in paths:
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                character = row["character"]
                if len(character) != 1:
                    continue
                frequency = int(row["total_frequency_all_time"])
                if frequency < min_frequency:
                    continue
                atom_type = inferred_type(
                    character, row["unicode_category"], row["unicode_name"]
                )
                if atom_type is None:
                    continue
                current_type = mapping.get(character)
                if current_type is not None:
                    # The legacy RUSH group mixes Latin diacritics with Greek
                    # and Cyrillic. Reclassify only frequent Latin characters
                    # so words such as "café" remain one English Atom.
                    if current_type == "RUSH" and atom_type == "ENG":
                        mapping[character] = "ENG"
                    continue
                previous = candidates.get(character)
                if previous is None or frequency > previous[0]:
                    candidates[character] = (frequency, atom_type)

    ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], ord(item[0])))
    if max_new_characters:
        ranked = ranked[:max_new_characters]
    for character, (_frequency, atom_type) in ranked:
        mapping[character] = atom_type
    return len(ranked)


def write_tmap(mapping, path, line_width=96):
    groups = {}
    for character, atom_type in mapping.items():
        groups.setdefault(atom_type, []).append(character)
    order = list(TYPE_ORDER)
    order.extend(sorted(set(groups) - set(order)))

    lines = []
    for atom_type in order:
        characters = sorted(set(groups.get(atom_type, ())), key=ord)
        if not characters:
            continue
        lines.append(f"-%{atom_type}")
        for start in range(0, len(characters), line_width):
            lines.append("".join(characters[start:start + line_width]))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_TMAP,
                        help="existing canonical tmap used as the update baseline")
    parser.add_argument("--finefreq", action="append", default=[],
                        help="FineFreq CSV; may be specified more than once")
    parser.add_argument("--output", default=DEFAULT_TMAP)
    parser.add_argument("--min-frequency", type=int, default=100_000)
    parser.add_argument("--max-new-characters", type=int, default=4096)
    args = parser.parse_args()

    mapping = read_tmap(args.input)
    baseline_count = len(mapping)
    added = add_finefreq(
        mapping, args.finefreq, args.min_frequency, args.max_new_characters
    )
    write_tmap(mapping, Path(args.output))
    print(
        f"tmap={args.output} baseline={baseline_count} finefreq={added} "
        f"total={len(mapping)} bytes={Path(args.output).stat().st_size}"
    )


if __name__ == "__main__":
    main()
