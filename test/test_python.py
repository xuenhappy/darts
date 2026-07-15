import concurrent.futures
from pathlib import Path
import tempfile
import unittest

from darts import Dregex, DSegment, PyAtomList


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "conf.json"


class AtomListTests(unittest.TestCase):
    def test_atom_types_and_boundaries(self):
        atoms = PyAtomList("中文ABC123").tolist()
        self.assertEqual([atom.image for atom in atoms], ["中", "文", "ABC", "123"])
        self.assertEqual([(atom.st, atom.et) for atom in atoms], [(0, 1), (1, 2), (2, 5), (5, 8)])

    def test_normalization_and_spaces(self):
        atoms = PyAtomList("ＡＢＣ １２３", skip_space=True, normal_before=True).tolist()
        self.assertEqual([atom.image for atom in atoms], ["ABC", "123"])

    def test_unicode_character_types(self):
        self.assertEqual(PyAtomList("あ").tolist()[0].chtype, "WUNK")
        self.assertEqual(PyAtomList("한").tolist()[0].chtype, "WUNK")
        self.assertEqual(PyAtomList("𠀀").tolist()[0].chtype, "CJK")

    def test_thread_safe_atomization(self):
        expected = [atom.image for atom in PyAtomList("并发中文ABC123").tolist()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(
                lambda _: [atom.image for atom in PyAtomList("并发中文ABC123").tolist()], range(500)
            ))
        self.assertTrue(all(result == expected for result in results))


class DictionaryTests(unittest.TestCase):
    def test_compile_casefold_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "test.pbs")
            Dregex.compile(path, iter([
                (["中", "文"], ["LANG"]),
                (["abc", "123"], ["MIXED"]),
            ]))
            regex = Dregex(path)
            hits = []
            regex.parse(["中", "文", "ABC", "123"],
                        lambda start, end, labels: hits.append((start, end, set(labels))) or False)
            self.assertEqual(hits[0], (0, 2, {"LANG"}))
            self.assertEqual(hits[1], (2, 4, {"MIXED"}))


class SegmentTests(unittest.TestCase):
    def assert_contiguous_cover(self, mode, text):
        atoms, output = DSegment(str(CONFIG), mode).cut(text)
        words = output.tolist()
        self.assertTrue(words)
        self.assertEqual(words[0].atom_s, 0)
        self.assertEqual(words[-1].atom_e, len(atoms))
        self.assertTrue(all(left.atom_e == right.atom_s for left, right in zip(words, words[1:])))

    def test_non_neural_modes(self):
        text = "目标检测模型量化和中文分词测试"
        self.assert_contiguous_cover("faster", text)
        self.assert_contiguous_cover("fast", text)
        self.assert_contiguous_cover("hybrid", text)
        self.assert_contiguous_cover("", text)

    def test_long_text(self):
        self.assert_contiguous_cover("faster", "中文分词测试。" * 40)

    def test_pinyin_mode_and_single_character(self):
        _atoms, output = DSegment(str(CONFIG), "pinyin").cut("重庆音乐ABC")
        words = output.tolist()
        self.assertIn("chóng qìng", next(word for word in words if word.image == "重庆").labels)
        self.assertIn("yīn yuè", next(word for word in words if word.image == "音乐").labels)
        self.assertFalse(next(word for word in words if word.image == "ABC").labels - {"ENG"})

        _atoms, output = DSegment(str(CONFIG), "pinyin").cut("中")
        self.assertIn("zhōng", output.tolist()[0].labels)

    def test_missing_mode_fails(self):
        with self.assertRaises(OSError):
            DSegment(str(CONFIG), "missing-mode")


if __name__ == "__main__":
    unittest.main()
