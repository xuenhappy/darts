"""Darts part-of-speech labels and mappings to common open tag sets."""

from dataclasses import dataclass
from typing import Iterable


# Darts uses namespaced labels internally so character type ``POS`` cannot be
# confused with part-of-speech annotations. The external short names follow
# the widely used LAC/jieba convention where a reliable mapping exists.
_DARTS_TO_SHORT = {
    "POS_ADJ": "a", "POS_ADP": "p", "POS_ADV": "d", "POS_AUX": "u",
    "POS_CCONJ": "c", "POS_DET": "r", "POS_NOUN": "n", "POS_NUM": "m",
    "POS_PART": "u", "POS_PRON": "r", "POS_PROPN": "nz", "POS_PUNCT": "w",
    "POS_SCONJ": "c", "POS_SYM": "w", "POS_VERB": "v", "POS_X": "xc",
    "POS_n": "n", "POS_f": "f", "POS_s": "s", "POS_t": "t",
    "POS_nr": "nr", "POS_ns": "ns", "POS_nt": "nt", "POS_nw": "nw",
    "POS_nz": "nz", "POS_v": "v", "POS_vd": "vd", "POS_vn": "vn",
    "POS_a": "a", "POS_ad": "ad", "POS_an": "an", "POS_d": "d",
    "POS_m": "m", "POS_q": "q", "POS_r": "r", "POS_p": "p",
    "POS_c": "c", "POS_u": "u", "POS_xc": "xc", "POS_w": "w",
}

_CHAR_FALLBACK = {
    "CJK": "xc", "ENG": "xc", "NUM": "m", "POS": "w", "SYMBOLS": "w",
    "DATE": "t", "DIGIT": "m",
}


def select_pos(labels: Iterable[str]) -> str:
    """Return one stable LAC-like POS tag from a runtime label collection."""
    candidates = sorted(label for label in labels if label.startswith("POS_"))
    if candidates:
        return _DARTS_TO_SHORT.get(candidates[0], candidates[0][4:].lower())
    for label in ("DATE", "DIGIT", "NUM", "POS", "SYMBOLS", "ENG", "CJK"):
        if label in labels:
            return _CHAR_FALLBACK[label]
    return "xc"


@dataclass(frozen=True)
class LacToken:
    text: str
    pos: str
    start: int
    end: int
    labels: frozenset

