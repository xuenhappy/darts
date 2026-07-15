import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest

from darts import (Dregex, DSegment, LocationSegmenter, PinyinAnnotator, PyAtomList,
                   Tokenizer, sentence_pinyin)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "conf.json"


class AtomListTests(unittest.TestCase):
    def test_atom_types_and_boundaries(self):
        atoms = PyAtomList("中文ABC123").tolist()
        self.assertEqual([atom.image for atom in atoms], ["中", "文", "ABC", "123"])
        self.assertEqual([(atom.st, atom.et) for atom in atoms], [(0, 1), (1, 2), (2, 5), (5, 8)])

    def test_native_lists_support_safe_indexing(self):
        atom_list = PyAtomList("中文ABC123")
        self.assertEqual(atom_list[0].image, "中")
        self.assertEqual(atom_list[0].chtype, "CJK")
        with self.assertRaises(IndexError):
            _ = atom_list[len(atom_list)]

        _atoms, words = DSegment(str(CONFIG), "hybrid").cut("中文分词")
        self.assertEqual(words[0].image, words.tolist()[0].image)
        with self.assertRaises(IndexError):
            _ = words[len(words)]

    def test_normalization_and_spaces(self):
        atoms = PyAtomList("ＡＢＣ １２３", skip_space=True, normal_before=True).tolist()
        self.assertEqual([atom.image for atom in atoms], ["ABC", "123"])
        atoms = PyAtomList("ABC ， DEF", skip_space=True, normal_before=False).tolist()
        self.assertEqual([atom.image for atom in atoms], ["ABC", "，", "DEF"])
        self.assertNotIn(" ", "".join(atom.image for atom in atoms))

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

    def test_cut_exposes_training_atomization_options(self):
        atoms, _words = DSegment(str(CONFIG), "faster").cut(
            "ＡＢＣ １２３", skip_space=True, normal_before=False
        )
        self.assertEqual([atom.image for atom in atoms.tolist()], ["ＡＢＣ", "１２３"])

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

    def test_only_selected_mode_dependency_graph_is_instantiated(self):
        config = {
            "dservices": {
                "unused.service": {"type": "TypeThatDoesNotExist"},
            },
            "recognizers": {
                "unused.recognizer": {
                    "type": "DictWordRecongnizer",
                    "pbfile.path": "/definitely/missing.pbs",
                    "deps": {"unused": "unused.service"},
                },
            },
            "deciders": {
                "selected.decider": {"type": "MinCoverDecider"},
                "unused.decider": {"type": "TypeThatDoesNotExist"},
            },
            "modes": {
                "selected": {"decider": "selected.decider", "recognizers": []},
                "broken": {"decider": "unused.decider", "recognizers": ["unused.recognizer"]},
            },
            "default.mode": "selected",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lazy.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            segment = DSegment(str(path), "selected")
            atoms, words = segment.cut("中文")
            self.assertEqual(len(atoms), len(words))
            with self.assertRaises(OSError):
                DSegment(str(path), "broken")


class PinyinTests(unittest.TestCase):
    def setUp(self):
        self.annotator = PinyinAnnotator(str(CONFIG))

    def test_sentence_annotation_disambiguates_polyphonic_words(self):
        sentence = "重庆音乐ABC"
        tokens = self.annotator.annotate(sentence)
        self.assertEqual([token.text for token in tokens], ["重庆", "音乐", "ABC"])
        self.assertEqual([token.pinyin for token in tokens], ["chóng qìng", "yīn yuè", None])
        self.assertEqual([(token.start, token.end) for token in tokens], [(0, 2), (2, 4), (4, 7)])
        self.assertTrue(all(sentence[token.start:token.end] == token.text for token in tokens))

        readings = self.annotator.readings("银行行长")
        self.assertEqual(readings, ["yín háng", "háng zhǎng"])

    def test_non_cjk_policy_and_sentence_format(self):
        self.assertEqual(
            self.annotator.readings("中国AI2026", non_cjk="<NO_PINYIN>"),
            ["zhōng guó", "<NO_PINYIN>", "<NO_PINYIN>"],
        )
        self.assertEqual(self.annotator.format("重庆ABC"), "chóng qìng ABC")
        self.assertEqual(
            self.annotator.format("重庆ABC", preserve_non_cjk=False),
            "chóng qìng",
        )
        self.assertEqual(sentence_pinyin("中"), "zhōng")

    def test_empty_and_invalid_sentence(self):
        self.assertEqual(self.annotator.annotate(""), [])
        with self.assertRaises(TypeError):
            self.annotator.annotate(None)
        with self.assertRaises(TypeError):
            self.annotator.annotate("中文", non_cjk=1)


class LocationTests(unittest.TestCase):
    def setUp(self):
        self.segmenter = LocationSegmenter(str(CONFIG))

    def test_dictionary_rules_and_role_quantizer_form_address_path(self):
        address = "浙江省杭州市西湖区文三路90号东部软件园2号楼"
        result = self.segmenter.parse(address)
        self.assertEqual(
            [(token.text, token.kind) for token in result.tokens],
            [("浙江省", "province"), ("杭州市", "city"), ("西湖区", "district"),
             ("文三路", "road"), ("90号", "component"), ("东部软件园", "poi"),
             ("2号楼", "component")],
        )
        self.assertTrue(all(address[token.start:token.end] == token.text
                            for token in result.tokens))
        self.assertEqual(result.poi, "东部软件园")

    def test_street_and_oov_rule_candidates_do_not_swallow_hierarchy(self):
        result = self.segmenter.parse("北京市海淀区中关村街道科学院南路88号")
        self.assertEqual(
            [(token.text, token.kind) for token in result.tokens],
            [("北京市", "province"), ("海淀区", "district"), ("中关村街道", "street"),
             ("科学院南路", "road"), ("88号", "component")],
        )

    def test_empty_and_invalid_address(self):
        self.assertEqual(self.segmenter.parse("").tokens, ())
        with self.assertRaises(TypeError):
            self.segmenter.parse(None)


class TokenizerTests(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer(str(CONFIG), "hybrid")

    def test_common_cut_interfaces(self):
        text = "目标检测模型量化"
        words = self.tokenizer.lcut(text)
        self.assertEqual("".join(words), text)
        self.assertEqual(list(self.tokenizer.cut(text)), words)
        self.assertEqual(self.tokenizer(text), words)
        self.assertEqual(self.tokenizer.batch_cut([text, text]), [words, words])

    def test_offsets_and_labels(self):
        text = "中文ABC分词"
        tuples = list(self.tokenizer.tokenize(text))
        self.assertEqual("".join(token for token, _start, _end in tuples), text)
        self.assertTrue(all(text[start:end] == token for token, start, end in tuples))
        rich = self.tokenizer.tokens(text)
        self.assertEqual([(item.text, item.start, item.end) for item in rich], tuples)

    def test_empty_and_invalid_input(self):
        self.assertEqual(self.tokenizer.lcut(""), [])
        with self.assertRaises(TypeError):
            self.tokenizer.lcut(None)
        with self.assertRaises(ValueError):
            list(self.tokenizer.tokenize("中文", mode="invalid"))


if __name__ == "__main__":
    unittest.main()
