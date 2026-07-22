import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).parents[1] / "scripts" / "extract_wikimedia.py"
SPEC = importlib.util.spec_from_file_location("extract_wikimedia", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExtractWikimediaTest(unittest.TestCase):
    def test_clean_wikitext_retains_visible_text(self):
        source = "'''南京市'''是[[江苏省|江苏]]省会。{{cite web|url=x}}\n\n== 历史 ==\n建城已久。"
        self.assertEqual(MODULE.clean_wikitext(source), ["南京市是江苏省会。", "历史 建城已久。"]) 


if __name__ == "__main__":
    unittest.main()
