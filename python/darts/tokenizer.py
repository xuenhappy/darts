"""High-level tokenizer API compatible with common Chinese tokenizer usage."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

from .cdarts import DSegment, PyWord
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
