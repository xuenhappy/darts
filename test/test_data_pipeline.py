import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("data_pipeline", ROOT / "scripts" / "data_pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class DataPipelineTest(unittest.TestCase):
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
