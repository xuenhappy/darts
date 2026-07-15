"""Structured Chinese administrative-address and POI segmentation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cdarts import DSegment


_LABEL_KIND = {
    "ADDR_PROVINCE": "province",
    "ADDR_CITY": "city",
    "ADDR_DISTRICT": "district",
    "ADDR_STREET": "street",
    "ADDR_DIVISION": "division",
    "ADDR_ROAD": "road",
    "ADDR_COMPONENT": "component",
    "ADDR_POI": "poi",
}
_KIND_PRIORITY = ("province", "city", "district", "street", "road", "component", "poi", "division")


@dataclass(frozen=True)
class LocationToken:
    """One address token with its semantic role and original character range."""

    text: str
    kind: str
    start: int
    end: int
    labels: frozenset


@dataclass(frozen=True)
class LocationResult:
    """A complete address segmentation and grouped semantic components."""

    text: str
    tokens: Tuple[LocationToken, ...]
    components: Dict[str, Tuple[str, ...]]

    def values(self, kind: str) -> Tuple[str, ...]:
        return self.components.get(kind, ())

    @property
    def poi(self) -> Optional[str]:
        values = self.values("poi")
        return values[-1] if values else None


class LocationSegmenter:
    """Reusable address segmenter backed by address-specific runtime plugins.

    The configured dictionary recognizes known administrative names and POIs;
    ``AddressRecongnizer`` supplies bounded suffix and numeric rules for unseen
    roads, buildings, house numbers and units. ``AddressDecider`` then selects a
    path using role-transition negative log probabilities plus address Bigram
    evidence.
    """

    def __init__(self, config: str = "data/conf.json"):
        self.config = config
        self._segment = DSegment(config, "location")

    @staticmethod
    def _kind(labels) -> str:
        kinds = {_LABEL_KIND[label] for label in labels if label in _LABEL_KIND}
        return next((kind for kind in _KIND_PRIORITY if kind in kinds), "unknown")

    def parse(self, address: str) -> LocationResult:
        """Segment one address and retain offsets into the original string."""
        if not isinstance(address, str):
            raise TypeError("address must be str")
        if not address:
            return LocationResult(address, (), {})
        atoms, words = self._segment.cut(address)
        atom_values = atoms.tolist()
        tokens = []
        grouped = {}
        for word in words.tolist():
            if word.atom_s < 0 or word.atom_e <= word.atom_s or word.atom_e > len(atom_values):
                raise ValueError(f"invalid word atom range [{word.atom_s}, {word.atom_e})")
            start = atom_values[word.atom_s].st
            end = atom_values[word.atom_e - 1].et
            kind = self._kind(word.labels)
            token = LocationToken(word.image, kind, start, end, frozenset(word.labels))
            tokens.append(token)
            grouped.setdefault(kind, []).append(word.image)
        components = {kind: tuple(values) for kind, values in grouped.items()}
        return LocationResult(address, tuple(tokens), components)

    def lcut(self, address: str) -> List[str]:
        return [token.text for token in self.parse(address).tokens]

    def tokenize(self, address: str) -> List[Tuple[str, str, int, int]]:
        """Return ``(text, kind, start, end)`` tuples for serialization."""
        return [(token.text, token.kind, token.start, token.end)
                for token in self.parse(address).tokens]

    def __call__(self, address: str) -> LocationResult:
        return self.parse(address)


_default_segmenter = None


def parse_location(address: str) -> LocationResult:
    """Parse an address with a lazily initialized process-wide segmenter."""
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = LocationSegmenter()
    return _default_segmenter.parse(address)
