"""
File names as people actually save them.

The same document arrives as 'Microsoft_CFO_transition.pdf' from one export and
'Microsoft CFO transition.pdf' from another; the spreadsheets as
'FX_rates.xlsx' or 'FX rates.xlsx'. Code that looks a file up by name compares
keys that ignore case, spacing and punctuation, so a rename like that never
makes a file disappear.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def name_key(name: str) -> str:
    """'Banque Léman - relevé.pdf' -> 'banquelemanrelevepdf'."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def same_name(a: str, b: str) -> bool:
    return name_key(a) == name_key(b)


def resolve(folder: str | Path, name: str) -> Path:
    """The file in `folder` that `name` refers to, however it was spaced.

    Falls back to folder/name, so a genuinely missing file still fails loudly
    where it is opened rather than here.
    """
    folder = Path(folder)
    exact = folder / name
    if exact.exists():
        return exact
    key = name_key(name)
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if name_key(path.name) == key:
                return path
    return exact
