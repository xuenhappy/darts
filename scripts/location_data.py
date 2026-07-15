#!/usr/bin/env python3
"""Build address-specialized dictionary and transition models from open data."""

import argparse
from collections import Counter
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/external/province-city-china/data.csv"
DEFAULT_POI = ROOT / "data/external/location/poi.txt"
DEFAULT_OUTPUT = ROOT / "data/models"
DEFAULT_WORK = ROOT / "data/generated/location"

LEVEL_LABELS = {
    1: "ADDR_PROVINCE",
    2: "ADDR_CITY",
    3: "ADDR_DISTRICT",
    4: "ADDR_STREET",
}
ADDRESS_TERMS = {
    "街道": "ADDR_ROAD", "大道": "ADDR_ROAD", "公路": "ADDR_ROAD",
    "路口": "ADDR_ROAD", "胡同": "ADDR_ROAD", "小区": "ADDR_POI",
    "大厦": "ADDR_POI", "广场": "ADDR_POI", "中心": "ADDR_POI", "园区": "ADDR_POI",
    "公园": "ADDR_POI", "车站": "ADDR_POI", "机场": "ADDR_POI",
    "医院": "ADDR_POI", "学校": "ADDR_POI", "大学": "ADDR_POI",
}


def level_of(row):
    name = row.get("name", "")
    if row.get("town"):
        return 4
    if name.endswith(("街道", "镇", "乡", "苏木", "民族乡", "办事处")):
        return 4
    if row.get("area"):
        return 3
    if row.get("city"):
        return 2
    return 1


def load_divisions(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("name") and "-" not in row["name"]]
    return rows


def hierarchy_indexes(rows):
    indexes = ({}, {}, {})
    for row in rows:
        level = level_of(row)
        if level == 1:
            indexes[0][row["province"]] = row["name"].strip()
        elif level == 2:
            indexes[1][(row["province"], row["city"])] = row["name"].strip()
        elif level == 3:
            indexes[2][(row["province"], row["city"], row["area"])] = row["name"].strip()
    return indexes


def parent_names(row, indexes):
    province, city, district = indexes
    keys = (
        province.get(row["province"]),
        city.get((row["province"], row["city"])),
        district.get((row["province"], row["city"], row["area"])),
    )
    path = [name for name in keys if name]
    if level_of(row) == 4:
        path.append(row["name"].strip())
    return list(dict.fromkeys(path))


def load_poi(path):
    if not path or not path.exists():
        return []
    result = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            name = line.strip().split("\t", 1)[0]
            if len(name) >= 2:
                result.append(name)
    return result


def build(args):
    from darts import build_gramdict_fromfile, compileDregex

    rows = load_divisions(Path(args.source))
    indexes = hierarchy_indexes(rows)
    entries = {}
    singles = Counter()
    bigrams = Counter()

    for row in rows:
        name = row["name"].strip()
        entries.setdefault(name, set()).add(LEVEL_LABELS[level_of(row)])
        singles[name] += 10
        path = parent_names(row, indexes)
        bigrams.update(zip(path, path[1:]))

    for name, label in ADDRESS_TERMS.items():
        entries.setdefault(name, set()).add(label)
        singles[name] += 5
    for name in load_poi(Path(args.poi) if args.poi else None):
        entries.setdefault(name, set()).add("ADDR_POI")
        singles[name] += 3

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    text_dict = work / "location-dictionary.txt"
    single_file = work / "location-single.txt"
    union_file = work / "location-union.txt"
    with open(text_dict, "w", encoding="utf-8") as stream:
        for name in sorted(entries):
            stream.write(f"{','.join(sorted(entries[name]))}:{name}\n")
    with open(single_file, "w", encoding="utf-8") as stream:
        for name, count in sorted(singles.items()):
            stream.write(f"{name}\t{count}\n")
    with open(union_file, "w", encoding="utf-8") as stream:
        for (left, right), count in sorted(bigrams.items()):
            stream.write(f"{left}-{right}\t{count}\n")

    compileDregex(((name, ",".join(sorted(labels))) for name, labels in entries.items()),
                  str(output / "location.pbs"))
    build_gramdict_fromfile(str(single_file), str(union_file), str(output / "location.bdf"))
    print(f"divisions={len(rows)} dictionary={len(entries)} transitions={len(bigrams)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--poi", default=str(DEFAULT_POI))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.set_defaults(func=build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
