"""Card cost database: loading, name normalisation and fuzzy lookup."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

DEFAULT_CARDS_PATH = Path(__file__).parent / "data" / "cards.json"
MIRROR = "mirror"


def normalize(name: str) -> str:
    """'Mini P.E.K.K.A' -> 'mini_pekka', 'X-Bow' -> 'x_bow'."""
    name = name.strip().lower().replace(".", "")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


class CardDB:
    def __init__(self, costs: dict[str, int | str]):
        self.costs = {normalize(k): v for k, v in costs.items() if not k.startswith("_")}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CardDB":
        with open(path or DEFAULT_CARDS_PATH, encoding="utf-8") as f:
            return cls(json.load(f))

    def __contains__(self, name: str) -> bool:
        return normalize(name) in self.costs

    def cost(self, name: str) -> int | str:
        """Elixir cost, or the string 'mirror' for Mirror. Raises KeyError if unknown."""
        return self.costs[normalize(name)]

    def resolve(self, text: str) -> str | None:
        """Turn user input into a card key: exact match, unique prefix, then fuzzy match."""
        key = normalize(text)
        if not key:
            return None
        if key in self.costs:
            return key
        prefixed = [k for k in self.costs if k.startswith(key)]
        if len(prefixed) == 1:
            return prefixed[0]
        close = difflib.get_close_matches(key, self.costs.keys(), n=1, cutoff=0.75)
        return close[0] if close else None

    def suggestions(self, text: str, n: int = 5) -> list[str]:
        key = normalize(text)
        prefixed = sorted(k for k in self.costs if k.startswith(key))
        return (prefixed or difflib.get_close_matches(key, self.costs.keys(), n=n, cutoff=0.4))[:n]


def pretty(key: str) -> str:
    return key.replace("_", " ").title()
