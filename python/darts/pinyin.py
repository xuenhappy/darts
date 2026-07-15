"""Sentence-level, phrase-disambiguated Chinese pinyin annotation."""

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence, Union

from .cdarts import DSegment


# Runtime labels also contain values such as CJK, JIEBA_n and PROPN.  Pinyin
# labels are lower-case syllables, optionally separated by spaces and carrying
# tone marks.  Keeping this filter here avoids coupling the public API to the
# internal numeric PinyinEncoder codes.
_PINYIN_RE = re.compile(
    r"[a-züêāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ]+"
    r"(?:\s+[a-züêāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ]+)*$"
)


@dataclass(frozen=True)
class PinyinToken:
    """One segmented sentence token and its context-disambiguated reading.

    ``start`` and ``end`` are half-open Python character offsets into the
    original sentence. ``pinyin`` is ``None`` for non-Chinese tokens unless a
    caller-provided replacement is used.
    """

    text: str
    pinyin: Optional[str]
    start: int
    end: int
    labels: frozenset


class PinyinAnnotator:
    """Annotate complete sentences using Darts' phrase-aware pinyin mode.

    The underlying segmenter first selects a word path, then looks up phrase
    readings to resolve polyphonic characters. Unknown phrases fall back to
    character readings in the C++ recognizer. One annotator should be reused so
    dictionaries and configuration are loaded only once.
    """

    def __init__(self, config: str = "data/conf.json"):
        self.config = config
        self._segment = DSegment(config, "pinyin")

    @staticmethod
    def _reading(labels: Sequence[str]) -> Optional[str]:
        readings = sorted(label for label in labels if _PINYIN_RE.fullmatch(label))
        # Phrase data should provide one reading after contextual
        # disambiguation. Sorting keeps malformed multi-label data deterministic.
        return readings[0] if readings else None

    def annotate(self, sentence: str, non_cjk: Optional[str] = None) -> List[PinyinToken]:
        """Return aligned word-level annotations for an entire sentence.

        Args:
            sentence: Input Unicode sentence.
            non_cjk: Optional replacement for tokens without Chinese pinyin.
                The default ``None`` explicitly represents "no pinyin".
        """
        if not isinstance(sentence, str):
            raise TypeError("sentence must be str")
        if non_cjk is not None and not isinstance(non_cjk, str):
            raise TypeError("non_cjk must be str or None")
        if not sentence:
            return []

        atoms, words = self._segment.cut(sentence)
        atom_values = atoms.tolist()
        result = []
        for word in words.tolist():
            if word.atom_s < 0 or word.atom_e <= word.atom_s or word.atom_e > len(atom_values):
                raise ValueError(f"invalid word atom range [{word.atom_s}, {word.atom_e})")
            start = atom_values[word.atom_s].st
            end = atom_values[word.atom_e - 1].et
            reading = self._reading(word.labels)
            result.append(PinyinToken(
                text=word.image,
                pinyin=reading if reading is not None else non_cjk,
                start=start,
                end=end,
                labels=frozenset(word.labels),
            ))
        return result

    def readings(self, sentence: str, non_cjk: Optional[str] = None) -> List[Optional[str]]:
        """Return one reading per segmented word, preserving sentence alignment."""
        return [token.pinyin for token in self.annotate(sentence, non_cjk=non_cjk)]

    def format(self, sentence: str, separator: str = " ", preserve_non_cjk: bool = True,
               non_cjk: Optional[str] = None) -> str:
        """Render a sentence as pinyin while optionally preserving non-Chinese text."""
        if not isinstance(separator, str):
            raise TypeError("separator must be str")
        values = []
        for token in self.annotate(sentence, non_cjk=non_cjk):
            if token.pinyin is not None:
                values.append(token.pinyin)
            elif preserve_non_cjk:
                values.append(token.text)
        return separator.join(values)

    def __call__(self, sentence: str, non_cjk: Optional[str] = None) -> List[PinyinToken]:
        return self.annotate(sentence, non_cjk=non_cjk)


_default_annotator = None


def _default() -> PinyinAnnotator:
    global _default_annotator
    if _default_annotator is None:
        _default_annotator = PinyinAnnotator()
    return _default_annotator


def annotate(sentence: str, non_cjk: Optional[str] = None) -> List[PinyinToken]:
    """Annotate a sentence with the process-wide default pinyin model."""
    return _default().annotate(sentence, non_cjk=non_cjk)


def sentence_pinyin(sentence: str, separator: str = " ", preserve_non_cjk: bool = True,
                    non_cjk: Optional[str] = None) -> str:
    """Format a complete sentence using phrase-disambiguated pinyin."""
    return _default().format(sentence, separator, preserve_non_cjk, non_cjk)
