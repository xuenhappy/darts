import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "pseudo_corpus.py"
SPEC = importlib.util.spec_from_file_location("pseudo_corpus", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PseudoCorpusTest(unittest.TestCase):
    def test_atom_count_keeps_english_and_decimal_runs(self):
        self.assertEqual(MODULE.atom_count("中文 OpenAI-compatible 37.5%"), 5)

    def test_split_long_preserves_text_and_limit(self):
        text = "南京市长江大桥，OpenAI-compatible API延迟下降37.5%。继续测试。"
        parts = list(MODULE.split_long(text, 8))
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(MODULE.atom_count(part) <= 8 for part in parts))

    def test_annotation_requires_lossless_reconstruction(self):
        valid = [("南京市", "POS_ns"), ("长江大桥", "POS_nz")]
        self.assertIsNone(MODULE.validate_annotation("南京市长江大桥", valid, 3, 20))
        invalid = [("南京", "POS_ns"), ("大桥", "POS_nz")]
        self.assertEqual(
            MODULE.validate_annotation("南京市长江大桥", invalid, 3, 20), "alignment"
        )

    def test_corpus_texts_reconstructs_binary_and_lac(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "binary.txt"
            lac = Path(directory) / "lac.txt"
            binary.write_text("南京市 长江大桥\n", encoding="utf-8")
            lac.write_text("南京市/POS_ns 长江大桥/POS_nz\n", encoding="utf-8")
            self.assertEqual(list(MODULE.corpus_texts(binary)), ["南京市长江大桥"])
            self.assertEqual(list(MODULE.corpus_texts(lac)), ["南京市长江大桥"])

    def test_canonicalize_annotation_drops_skipped_whitespace(self):
        annotation = [("OpenAI ", "POS_nz"), (" ", "POS_PUNCT"), ("API", "POS_nz")]
        self.assertEqual(
            MODULE.canonicalize_annotation(annotation),
            [("OpenAI", "POS_nz"), ("API", "POS_nz")],
        )

    def test_compact_pos_mapping_is_lossless(self):
        for pos, short in MODULE.POS_TO_SHORT.items():
            self.assertEqual(MODULE.SHORT_TO_POS[short], pos)


if __name__ == "__main__":
    unittest.main()
