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

        商务现代简约风格。每个切分子句包含：3×n 可拖动原子条（st~et 合并
        单元格）、atom 索引 n×n 上三角候选词矩阵（同格多 cell 彩色区分，
        下半三角与无词处置灰）、节点上三角转移边矩阵（x1@x2 + 权重，最优
        路径绿色，不可见处置灰）、最终切分结果。
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
            "</style></head><body><div class=\"wrap\">",
            "<h1><span class=\"dot\"></span>darts 切分报告</h1>",
            f"<p class=\"meta\">mode <b>{esc(self.mode or 'default')}</b> · "
            f"temperature <b>{temp if temp is not None else 'deterministic'}</b> · "
            f"sub-sentences <b>{len(report['subs'])}</b></p>",
            "<div class=\"card\"><h2>原始句子</h2>",
            f"<div class=\"sentence\">{esc(sentence)}</div>",
            "<h3>规范化后（normalize）</h3>",
            f"<div class=\"sentence\">{esc(_normalize(sentence))}</div></div>",
        ]
        for si, sub in enumerate(report["subs"], 1):
            best = set(sub.get("best", []))
            best_seq = list(sub.get("best", []))
            best_edges = set(zip(best_seq, best_seq[1:]))
            n_atoms = len(sub["atoms"])
            parts.append("<div class=\"card\">")
            parts.append(f"<h2>子句 {si}/{len(report['subs'])}</h2>")
            parts.append(f"<div class=\"sentence\">{esc(sub['text'])}</div>")

            # ── AtomList：真·二维矩阵 4 行 × n 列，列可滚动（sticky 首列）──
            parts.append("<h3>AtomList 矩阵（4×n · 列横向滚动）</h3>"
                         "<div class=\"drag compact\"><table class=\"matrix\">"
                         "<tr><th class=\"stick\">行＼col</th>")
            for i in range(n_atoms):
                parts.append(f"<th class=\"num\">{i}</th>")
            parts.append("</tr>")
            rows = [
                ("atom", lambda im, t, s, e: f"<td class=\"acell\">{esc(im)}</td>"),
                ("type", lambda im, t, s, e: f"<td class=\"tcell\">{esc(t)}</td>"),
                ("st", lambda im, t, s, e: f"<td class=\"pcell\">{s}</td>"),
                ("et", lambda im, t, s, e: f"<td class=\"pcell\">{e}</td>"),
            ]
            for name, fmt in rows:
                parts.append(f"<tr><th class=\"stick\">{name}</th>")
                for image, ctype, st, et in sub["atoms"]:
                    parts.append(fmt(image, ctype, st, et))
                parts.append("</tr>")
            parts.append("</table></div>")

            # ── 候选词矩阵：atom 索引 n×n 上三角（行 st，列 et）──
            span_words = {}
            for node in sub["nodes"]:
                span_words.setdefault((node["st"], node["et"]), []).append(node)
            n_nom = sum(1 for node in sub["nodes"]
                        if any(lb.startswith("NEURAL_") for lb in node["labels"]))
            parts.append(f"<h3>识别词矩阵（行=st 列=et · 同格多 cell 彩色 · "
                         f"★最优路径 · ⚡虚线=神经提名 {n_nom}/{len(sub['nodes'])} · "
                         f"灰=不可见/无词）</h3>"
                         "<div class=\"drag\"><table class=\"matrix\"><tr><th>"
                         "st＼et</th>")
            for et in range(1, n_atoms + 1):
                parts.append(f"<th class=\"num\">{et}</th>")
            parts.append("</tr>")
            for st in range(n_atoms):
                parts.append(f"<tr><th class=\"num\">{st}</th>")
                for et in range(1, n_atoms + 1):
                    if et <= st:
                        parts.append("<td class=\"dead\">·</td>")
                        continue
                    cells = span_words.get((st, et))
                    if not cells:
                        parts.append("<td class=\"void\">—</td>")
                        continue
                    chips = []
                    for ci, node in enumerate(cells):
                        color = f"c{ci % 8}"
                        star = " best" if node["idx"] in best else ""
                        label_main = next(
                            (lb for lb in node["labels"]
                             if lb.startswith("POS_")), "")
                        fx = (f"<span class=\"fx\">#{node['idx']}"
                              f"{'′' + str(-node['feat']) if node['feat'] < -1 else ''}"
                              f"</span>")
                        neural = any(lb.startswith("NEURAL_")
                                     for lb in node["labels"])
                        chips.append(
                            f"<span class=\"wcell {color}{star}"
                            f"{' neural' if neural else ''}\" title="
                            f"\"idx={node['idx']} feat={node['feat']} "
                            f"labels={esc(','.join(node['labels']))}\">"
                            f"{esc(node['text'])}<small>{esc(label_main)}</small>"
                            f"{'<b class=\"nb\">⚡</b>' if neural else ''}"
                            f"{fx}</span>")
                    parts.append(f"<td>{''.join(chips)}</td>")
                parts.append("</tr>")
            parts.append("</table></div>")

            # ── 转移边矩阵：节点 idx 上三角（不可见处置灰）──
            edges = {(e["f"], e["t"]): e["w"] for e in sub["edges"]}
            node_ids = sorted({n["idx"] for n in sub["nodes"]})
            size = max(node_ids) + 1 if node_ids else 0
            ids = [-1] + node_ids + [size]
            short = {n["idx"]: (n["text"], n["st"], n["et"])
                     for n in sub["nodes"]}

            def nlabel(i):
                if i == -1:
                    return "␣START"
                if i == size:
                    return "END␣"
                w, s, e = short[i]
                return f"{w}[{s},{e})"
            parts.append("<h3>SegPath 转移边矩阵（节点上三角 · x1@x2 + 权重 · "
                         "最优路径绿色 · 灰=不可达）</h3><div class=\"drag\">"
                         "<table class=\"matrix\"><tr><th>from＼to</th>")
            for t in ids:
                tag = " best" if t in best else ""
                star = " ★" if tag else ""
                parts.append(f"<th class=\"corner{tag}\">{esc(nlabel(t))}{star}<br/>"
                             f"<small>#{t}</small></th>")
            parts.append("</tr>")
            for f in ids:
                tag = " best" if f in best else ""
                star = " ★" if tag else ""
                parts.append(f"<tr><th class=\"corner{tag}\">{esc(nlabel(f))}{star}<br/>"
                             f"<small>#{f}</small></th>")
                for t in ids:
                    if ids.index(t) <= ids.index(f):
                        parts.append("<td class=\"dead\">·</td>")
                        continue
                    if (f, t) in edges:
                        w = edges[(f, t)]
                        cls = (" class=\"edgecell bestedge\""
                               if (f, t) in best_edges
                               else " class=\"edgecell\"")
                        parts.append(
                            f"<td{cls}>{esc(nlabel(f))}<b>@</b>"
                            f"{esc(nlabel(t))}<br/>{w:.3f}</td>")
                    else:
                        parts.append("<td class=\"void\">—</td>")
                parts.append("</tr>")
            parts.append("</table></div>")

            # ── 最终结果 ──
            parts.append("<h3>最终输出</h3><div class=\"final\">")
            for word, labels in sub["out"]:
                tag = next((lb for lb in labels if lb.startswith("POS_")
                            and lb != "POS_PUNCT"), None)
                badge = (f"<span class=\"tag tag-hot\">{esc(tag)}</span>"
                         if tag else "")
                parts.append(f"<span class=\"w\"><b>{esc(word)}</b>{badge}</span>")
            parts.append("</div></div>")
        parts.append("</div>"
                     "<script>"
                     "for(const d of document.querySelectorAll('.drag'))"
                     "{let x=0,l=0,down=false;"
                     "d.addEventListener('mousedown',e=>{down=true;x=e.clientX;l=d.scrollLeft});"
                     "d.addEventListener('mouseleave',()=>down=false);"
                     "d.addEventListener('mouseup',()=>down=false);"
                     "d.addEventListener('mousemove',e=>{if(!down)return;"
                     "d.scrollLeft=l-(e.clientX-x)});}"
                     "</script></body></html>")
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
