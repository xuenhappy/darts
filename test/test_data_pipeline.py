import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("data_pipeline", ROOT / "scripts" / "data_pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class DataPipelineTest(unittest.TestCase):
    @staticmethod
    def demo_lines(name):
        return [
            line.strip()
            for line in (ROOT / "data" / "demo" / name).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_cws_demo_is_valid_and_train_dev_are_isolated(self):
        def read_demo(name):
            samples = [line.split() for line in self.demo_lines(name)]
            self.assertTrue(samples)
            self.assertTrue(all(len(tokens) >= 2 for tokens in samples))
            return samples

        train = read_demo("cws-train.txt")
        dev = read_demo("cws-dev.txt")
        train_texts = {"".join(tokens) for tokens in train}
        dev_texts = {"".join(tokens) for tokens in dev}

        self.assertTrue(train_texts.isdisjoint(dev_texts))
        self.assertTrue(any(any(2 <= len(token) <= 5 for token in row) for row in train))
        self.assertTrue(any(any(len(token) > 5 for token in row) for row in train))
        self.assertTrue(any(any(not token.isalpha() for token in row) for row in train))

    def test_lac_demo_uses_known_pos_labels_and_matches_cws_topics(self):
        pos_labels = {
            line.split("#", 1)[0]
            for line in (ROOT / "data" / "codes" / "pos.hx.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }

        def read_lac(name):
            sentences = []
            for line in self.demo_lines(name):
                words = []
                for item in line.split():
                    word, separator, label = item.rpartition("/")
                    self.assertTrue(separator and word and label, item)
                    self.assertIn(label, pos_labels)
                    words.append(word)
                sentences.append("".join(words))
            return sentences

        lac_train = set(read_lac("lac-train.txt"))
        lac_dev = set(read_lac("lac-dev.txt"))
        cws_train = {"".join(line.split()) for line in self.demo_lines("cws-train.txt")}
        cws_dev = {"".join(line.split()) for line in self.demo_lines("cws-dev.txt")}
        self.assertTrue(lac_train <= cws_train)
        self.assertTrue(lac_dev <= cws_dev)
        self.assertTrue(lac_train.isdisjoint(lac_dev))

    def test_demo_dictionaries_and_bigram_files_are_consistent(self):
        type_labels = {
            line.split("#", 1)[0]
            for line in (ROOT / "data" / "codes" / "type.hx.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        pos_labels = {
            line.split("#", 1)[0]
            for line in (ROOT / "data" / "codes" / "pos.hx.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }

        def dictionary(name, allowed):
            words = set()
            for line in self.demo_lines(name):
                labels, separator, word = line.partition(":")
                self.assertTrue(separator and word, line)
                parsed = {
                    label.strip("<>")
                    for label in labels.split(",")
                    if label
                }
                self.assertTrue(parsed)
                self.assertTrue(parsed <= allowed, (name, parsed - allowed))
                words.add(word)
            return words

        general_words = dictionary("dregex_pattern_file.txt", type_labels)
        lac_words = dictionary("lac-dictionary.txt", pos_labels)
        self.assertIn("中文分词", general_words & lac_words)

        singles = {}
        for line in self.demo_lines("bigram_persenter_freq.txt"):
            columns = line.split("\t")
            self.assertGreaterEqual(len(columns), 2)
            singles[columns[0]] = int(columns[1])
            self.assertGreater(singles[columns[0]], 0)
        for line in self.demo_lines("bigram_persenter_freqR.txt"):
            pair, frequency = line.split("\t")
            left, separator, right = pair.partition("-")
            self.assertTrue(separator and left and right)
            self.assertIn(left, singles)
            self.assertIn(right, singles)
            self.assertGreater(int(frequency), 0)

    def test_specialized_demo_formats(self):
        for line in self.demo_lines("pinyin-phrases.txt"):
            word, separator, readings = line.partition(":")
            self.assertTrue(separator and word and readings.strip())
            self.assertEqual(len(word), len(readings.split()))

        address_labels = {
            "ADDR_PROVINCE", "ADDR_CITY", "ADDR_DISTRICT", "ADDR_STREET", "ADDR_POI"
        }
        for line in self.demo_lines("location-dictionary.txt"):
            label, separator, word = line.partition(":")
            self.assertTrue(separator and word)
            self.assertIn(label.strip("<>"), address_labels)

        temporal = self.demo_lines("temporal-quantity.txt")
        self.assertGreaterEqual(len(temporal), 5)
        self.assertTrue(any("2026年" in line for line in temporal))
        self.assertTrue(any("GB" in line for line in temporal))

    def test_read_conllu_skips_metadata_and_range_tokens(self):
        content = """# text = 我爱自然语言
1\t我\t_\tPRON\t_\t_\t_\t_\t_\t_
2-3\t自然语言\t_\t_\t_\t_\t_\t_\t_\t_
2\t自然\t_\tNOUN\t_\t_\t_\t_\t_\t_
3\t语言\t_\tNOUN\t_\t_\t_\t_\t_\t_

"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.conllu"
            source.write_text(content, encoding="utf-8")
            self.assertEqual(
                list(PIPELINE.read_conllu(source)),
                [[("我", "PRON"), ("自然", "NOUN"), ("语言", "NOUN")]],
            )

    def test_write_segmented_preserves_gold_boundaries(self):
        content = "1\t中文\t_\tNOUN\t_\t_\t_\t_\t_\t_\n2\t分词\t_\tNOUN\t_\t_\t_\t_\t_\t_\n\n"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.conllu"
            target = Path(directory) / "sample.txt"
            source.write_text(content, encoding="utf-8")
            self.assertEqual(PIPELINE.write_segmented(source, target), 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "中文 分词\n")


if __name__ == "__main__":
    unittest.main()
