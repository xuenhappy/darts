"""High-level tokenizer API compatible with common Chinese tokenizer usage."""

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

from .cdarts import DSegment, PyWord
from .cdarts import normalize as _normalize
from .pos import LacToken, select_pos


@dataclass(frozen=True)
class Token:
    """A token with original character offsets and immutable labels."""

    text: str
    start: int
    end: int
    labels: frozenset


class Tokenizer:
    """Reusable Darts tokenizer with jieba-like convenience methods.

    ``cut`` and ``lcut`` return strings for drop-in application use.
    ``tokenize`` returns ``(text, start, end)`` tuples like ``jieba.tokenize``;
    ``tokens`` exposes labels in structured :class:`Token` values.  ``cut_all``
    or ``mode="search"`` returns overlapping candidates and is not a unique
    segmentation path.

    Args:
        config: Runtime JSON configuration path.
        mode: Darts mode. ``None``/empty uses ``default.mode`` from the config.
    """

    def __init__(self, config: str = "data/conf.json", mode: str = ""):
        self.config = config
        self.mode = mode or ""
        self._segment = DSegment(config, self.mode)

    def _analyze(self, text: str, all_candidates: bool = False, temperature=None):
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not text:
            return [], []
        atoms, words = self._segment.cut(text, max_mode=all_candidates, temperature=temperature)
        return atoms.tolist(), words.tolist()

    @staticmethod
    def _offsets(atoms, word: PyWord) -> Tuple[int, int]:
        # Word positions are half-open Atom indexes; public offsets are
        # half-open character indexes in the original Python string.
        if word.atom_s < 0 or word.atom_e <= word.atom_s or word.atom_e > len(atoms):
            raise ValueError(f"invalid word atom range [{word.atom_s}, {word.atom_e})")
        return atoms[word.atom_s].st, atoms[word.atom_e - 1].et

    def cut(self, sentence: str, cut_all: bool = False, HMM: bool = True,
            use_paddle: bool = False) -> Iterator[str]:
        """Yield token strings; extra arguments match jieba's call signature.

        Darts chooses recognizers through configuration, so ``HMM`` and
        ``use_paddle`` are accepted for caller compatibility but do not alter the
        configured model.
        """
        del HMM, use_paddle
        _atoms, words = self._analyze(sentence, all_candidates=cut_all)
        return (word.image for word in words)

    def lcut(self, sentence: str, cut_all: bool = False, HMM: bool = True,
             use_paddle: bool = False) -> List[str]:
        """Return token strings as a list."""
        return list(self.cut(sentence, cut_all=cut_all, HMM=HMM, use_paddle=use_paddle))

    def tokens(self, sentence: str, all_candidates: bool = False) -> List[Token]:
        """Return structured tokens with offsets and labels."""
        atoms, words = self._analyze(sentence, all_candidates=all_candidates)
        result = []
        for word in words:
            start, end = self._offsets(atoms, word)
            result.append(Token(word.image, start, end, frozenset(word.labels)))
        return result

    def lac(self, sentence: str) -> List[LacToken]:
        """Return one segmented path with stable LAC-like POS tags."""
        atoms, words = self._analyze(sentence)
        result = []
        for word in words:
            start, end = self._offsets(atoms, word)
            labels = frozenset(word.labels)
            result.append(LacToken(word.image, select_pos(labels), start, end, labels))
        return result

    def sample(self, sentence: str, temperature: float = 0.5) -> List[str]:
        """Sample one complete segmentation from the Boltzmann path distribution."""
        _atoms, words = self._analyze(sentence, temperature=temperature)
        return [word.image for word in words]

    # ── seg_report：切分可视化 HTML 报告 ────────────────────────────
    _report_css_cache = None

    def _report_css(self) -> str:
        if Tokenizer._report_css_cache is None:
            css_path = Path(self.config).parent / "report.css"
            Tokenizer._report_css_cache = css_path.read_text(encoding="utf-8")
        return Tokenizer._report_css_cache

    def seg_report(self, sentence: str, temperature: float = None) -> str:
        """可视化切分报告 → HTML 字符串（CSS 固定于 data/report.css）。

        内容：原始/规范化句子、每个切分子句的原子表、候选词矩阵（词性）、
        转移边矩阵（x1@x2 + 权重，最优路径绿色）、最终切分结果。
        """
        if not isinstance(sentence, str) or not sentence:
            raise ValueError("sentence must be a non-empty string")
        temp = None if temperature is None else float(temperature)
        _atoms, _words, report_json = self._segment.cut_report(
            sentence, temperature=temp)
        report = json.loads(report_json)
        esc = html.escape

        parts = [
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">",
            "<title>darts 切分报告</title><style>", self._report_css(),
            "</style></head><body>",
            "<h1>darts 切分报告</h1>",
            f"<p class=\"meta\">mode={esc(self.mode or 'default')} · "
            f"temperature={temp if temp is not None else 'deterministic'} · "
            f"sub-sentences={len(report['subs'])}</p>",
            "<h3>原始句子</h3>",
            f"<div class=\"sentence\">{esc(sentence)}</div>",
            "<h3>规范化后句子</h3>",
            f"<div class=\"sentence\">{esc(_normalize(sentence))}</div>",
        ]
        for si, sub in enumerate(report["subs"], 1):
            best = set(sub.get("best", []))
            best_edges = {(f, t) for f, t in zip(
                sub.get("best", []), list(sub.get("best", []))[1:])}
            parts.append("<div class=\"subsection\">")
            parts.append(f"<h2>子句 {si}/{len(report['subs'])}</h2>")
            parts.append(f"<div class=\"sentence\">{esc(sub['text'])}</div>")

            # 原子表
            parts.append("<h3>AtomList（原子）</h3><table><tr>"
                         "<th>#</th><th>atom</th><th>type</th>"
                         "<th>st</th><th>et</th></tr>")
            for i, (image, ctype, st, et) in enumerate(sub["atoms"]):
                parts.append(
                    f"<tr><td class=\"num\">{i}</td><td>{esc(image)}</td>"
                    f"<td class=\"type\">{esc(ctype)}</td>"
                    f"<td class=\"num\">{st}</td><td class=\"num\">{et}</td></tr>")
            parts.append("</table>")

            # 候选词矩阵
            parts.append("<h3>识别词矩阵（候选格 · 词性标签）</h3>"
                         "<div class=\"scroll\"><table><tr><th>idx</th>"
                         "<th>word</th><th>[st,et)</th><th>feat</th>"
                         "<th>labels</th></tr>")
            for node in sub["nodes"]:
                cls = " class=\"best\"" if node["idx"] in best else ""
                tags = "".join(
                    f"<span class=\"tag{' tag-hot' if lb.startswith('POS_') else ''}\">"
                    f"{esc(lb)}</span>" for lb in node["labels"]) or "—"
                parts.append(
                    f"<tr{cls}><td class=\"num\">{node['idx']}</td>"
                    f"<td>{esc(node['text'])}</td>"
                    f"<td class=\"num\">[{node['st']},{node['et']})</td>"
                    f"<td class=\"num\">{node['feat']}</td><td>{tags}</td></tr>")
            parts.append("</table></div>")

            # 转移边矩阵（x1@x2 + 权重；最优路径边绿色）
            edges = {(e["f"], e["t"]): e["w"] for e in sub["edges"]}
            col_ids = sorted({t for (_f, t) in edges})
            row_ids = sorted({f for (f, _t) in edges})
            short = {n["idx"]: n["text"] for n in sub["nodes"]}

            def label(i):
                if i == -1:
                    return "␣"
                return short.get(i, "END")
            parts.append("<h3>SegPath 转移边矩阵（x1@x2 · 权重，最优路径绿色）"
                         "</h3><div class=\"scroll\"><table><tr><th>from＼to</th>")
            for t in col_ids:
                tl = "END" if t not in short else short[t]
                parts.append(f"<th>{esc(tl)}<br/><small>#{t}</small></th>")
            parts.append("</tr>")
            for f in row_ids:
                fl = "START" if f == -1 else short.get(f, str(f))
                parts.append(f"<tr><th>{esc(fl)}<br/><small>#{f}</small></th>")
                for t in col_ids:
                    if (f, t) in edges:
                        w = edges[(f, t)]
                        cls = " class=\"bestedge\"" if (f, t) in best_edges else ""
                        parts.append(
                            f"<td{cls}>{esc(str(label(f)))}@{esc(str(label(t)))}"
                            f"<br/><small>{w:.3f}</small></td>")
                    else:
                        parts.append("<td></td>")
                parts.append("</tr>")
            parts.append("</table></div>")

            # 最终结果
            parts.append("<h3>最终输出</h3><div class=\"final\">")
            for word, labels in sub["out"]:
                tag = next((lb for lb in labels if lb.startswith("POS_")
                            and lb != "POS_PUNCT"), None)
                badge = f" <span class=\"tag\">{esc(tag)}</span>" if tag else ""
                parts.append(f"<b>{esc(word)}</b>{badge}")
            parts.append("</div></div>")
        parts.append("</body></html>")
        return "".join(parts)

    def tokenize(self, unicode_sentence: str, mode: str = "default", HMM: bool = True
                 ) -> Iterator[Tuple[str, int, int]]:
        """Yield ``(token, start, end)`` tuples with jieba-compatible offsets."""
        del HMM
        if mode not in ("default", "search"):
            raise ValueError("mode must be 'default' or 'search'")
        return ((token.text, token.start, token.end)
                for token in self.tokens(unicode_sentence, all_candidates=mode == "search"))

    def batch_cut(self, sentences: Iterable[str], cut_all: bool = False) -> List[List[str]]:
        """Tokenize an iterable while reusing the loaded C++ segment instance."""
        return [self.lcut(sentence, cut_all=cut_all) for sentence in sentences]

    def __call__(self, value: Union[str, Sequence[str]], cut_all: bool = False):
        """Tokenize one string or a sequence of strings."""
        if isinstance(value, str):
            return self.lcut(value, cut_all=cut_all)
        return self.batch_cut(value, cut_all=cut_all)


_default_tokenizer = None


def _default() -> Tokenizer:
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()
    return _default_tokenizer


def cut(sentence: str, cut_all: bool = False, HMM: bool = True) -> Iterator[str]:
    """Module-level jieba-style tokenizer using the default Darts config."""
    return _default().cut(sentence, cut_all=cut_all, HMM=HMM)


def lcut(sentence: str, cut_all: bool = False, HMM: bool = True) -> List[str]:
    """List-returning variant of :func:`cut`."""
    return _default().lcut(sentence, cut_all=cut_all, HMM=HMM)


def tokenize(sentence: str, mode: str = "default", HMM: bool = True):
    """Module-level offset tokenizer."""
    return _default().tokenize(sentence, mode=mode, HMM=HMM)
