import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_tmap", ROOT / "scripts" / "build_tmap.py")
BUILD_TMAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_TMAP)


class CharacterMapTests(unittest.TestCase):
    def test_runtime_map_has_no_duplicate_effective_characters(self):
        mapping = BUILD_TMAP.read_tmap(ROOT / "data" / "kernel" / "chars.tmap")
        self.assertGreater(len(mapping), 1000)
        self.assertNotIn("中", mapping)
        self.assertEqual(mapping["😀"], "FACE")

    def test_finefreq_classification_preserves_cjk_boundaries(self):
        rows = [
            ("é", "Ll", "LATIN SMALL LETTER E WITH ACUTE", 1_000_000),
            ("٩", "Nd", "ARABIC-INDIC DIGIT NINE", 900_000),
            ("。", "Po", "IDEOGRAPHIC FULL STOP", 800_000),
            ("🧑", "So", "ADULT", 700_000),
            ("中", "Lo", "CJK UNIFIED IDEOGRAPH-4E2D", 2_000_000),
            ("한", "Lo", "HANGUL SYLLABLE HAN", 2_000_000),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "finefreq.csv"
            with open(source, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("character", "unicode_category", "unicode_name",
                                 "total_frequency_all_time"))
                writer.writerows(rows)
            mapping = {}
            added = BUILD_TMAP.add_finefreq(mapping, [source], 100_000, 0)
        self.assertEqual(added, 4)
        self.assertEqual(mapping["é"], "ENG")
        self.assertEqual(mapping["٩"], "NUM")
        self.assertEqual(mapping["。"], "POS")
        self.assertEqual(mapping["🧑"], "SYMBOLS")
        self.assertNotIn("中", mapping)
        self.assertNotIn("한", mapping)

    def test_frequent_latin_diacritic_corrects_legacy_rush_type(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "finefreq.csv"
            with open(source, "w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("character", "unicode_category", "unicode_name",
                                 "total_frequency_all_time"))
                writer.writerow(("é", "Ll", "LATIN SMALL LETTER E WITH ACUTE", 1_000_000))
            mapping = {"é": "RUSH", "Я": "RUSH"}
            BUILD_TMAP.add_finefreq(mapping, [source], 100_000, 0)
        self.assertEqual(mapping["é"], "ENG")
        self.assertEqual(mapping["Я"], "RUSH")

    def test_committed_map_is_canonical(self):
        mapping = BUILD_TMAP.read_tmap(ROOT / "data" / "kernel" / "chars.tmap")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chars.tmap"
            BUILD_TMAP.write_tmap(mapping, output)
            self.assertEqual(
                output.read_bytes(),
                (ROOT / "data" / "kernel" / "chars.tmap").read_bytes(),
            )

    def test_in_place_update_preserves_mapping_without_frequency_inputs(self):
        source = ROOT / "data" / "kernel" / "chars.tmap"
        expected = BUILD_TMAP.read_tmap(source)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chars.tmap"
            output.write_bytes(source.read_bytes())
            BUILD_TMAP.write_tmap(BUILD_TMAP.read_tmap(output), output)
            self.assertEqual(BUILD_TMAP.read_tmap(output), expected)


if __name__ == "__main__":
    unittest.main()
