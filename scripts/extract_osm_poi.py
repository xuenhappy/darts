#!/usr/bin/env python3
"""Stream Chinese POI names from an OpenStreetMap PBF into location_data input."""

import argparse
from pathlib import Path


POI_KEYS = ("amenity", "shop", "tourism", "leisure", "office", "craft", "healthcare")


def extract(args):
    try:
        import osmium
    except ImportError as error:
        raise RuntimeError("install the optional extractor dependency: pip install osmium") from error

    names = {}

    class Handler(osmium.SimpleHandler):
        def collect(self, tags):
            name = tags.get("name:zh") or tags.get("name")
            if not name or len(name.strip()) < 2:
                return
            category = next((f"{key}={tags[key]}" for key in POI_KEYS if key in tags), None)
            if category:
                names.setdefault(name.strip(), category)

        def node(self, item):
            self.collect(item.tags)

        def way(self, item):
            self.collect(item.tags)

        def relation(self, item):
            self.collect(item.tags)

    Handler().apply_file(args.input, locations=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        stream.write("# OpenStreetMap contributors; ODbL 1.0\n")
        for name, category in sorted(names.items()):
            stream.write(f"{name}\t{category}\n")
    print(f"poi={len(names)} output={output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Geofabrik or other .osm.pbf extract")
    parser.add_argument("--output", default="data/external/location/poi.txt")
    parser.set_defaults(func=extract)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
