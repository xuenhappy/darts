from .cdarts import (DSegment, Dregex, PyAtom, PyAtomList, PyWord, PyWordList, build_gramdict_fromfile, charDtype,
    AtomCodec, WordCodec)

from typing import Iterator, Tuple


def compileDregex(kvPiars: Iterator[Tuple[str, str]], outfile: str):
    """compile a double array NAF use k,v iterotor

    Args:
        kvPiars (Iterator[Tuple[str, str]]): _description_
        outfile (str): _description_
    """

    def dataiter():
        for k, v in kvPiars:
            if not k:
                continue
            k = [iterm.image for iterm in PyAtomList(k).tolist()]
            if len(k) < 2:
                continue
            v = [] if not v else [iterm.strip() for iterm in v.split(",")]
            yield (k, v)

    Dregex.compile(outfile, dataiter())
