"""Temporal perturbation: flip a before/after-style direction word, or shift
a date by +-1 year / +-1 quarter, so a temporally-blind embedding model can
be caught. Date shifts are format-preserving (zero-padded ISO stays
zero-padded) and clamp Feb 29 -> Feb 28 rather than raising, since
date(2024, 2, 29).replace(year=2025) raises ValueError.
"""
from __future__ import annotations

import random
import re
from datetime import date
from typing import List, Tuple

from ..text import Span
from ._dictionary import build_lookup, compile_phrase_regex, propose_swaps
from .base import Candidate

DIRECTION_PAIRS = [
    ("before", "after"),
    ("prior to", "following"),
    ("since", "until"),
    ("earlier", "later"),
    ("preceding", "subsequent"),
    ("past", "future"),
]

_DIRECTION_LOOKUP = build_lookup(DIRECTION_PAIRS)
_DIRECTION_RE = compile_phrase_regex(_DIRECTION_LOOKUP.keys())

_PREPOST_RE = re.compile(r"\b(pre|post)-(?=\w)", re.IGNORECASE)
_PREPOST_LOOKUP = {"pre": "post", "post": "pre"}

_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_YEAR_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\s+((?:19|20)\d{2})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _clamp_date(y: int, m: int, d: int) -> date:
    try:
        return date(y, m, d)
    except ValueError:
        if m == 2 and d == 29:
            return date(y, 2, 28)
        raise


class TemporalPerturbation:
    name = "temporal"
    kind = "temporal"
    priority = 40
    requires: Tuple[str, ...] = ()

    def __init__(self, max_per_text: int = 8):
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        out: List[Candidate] = []
        claimed: List[Tuple[int, int]] = []

        def _claim(start: int, end: int) -> bool:
            for s, e in claimed:
                if start < e and s < end:
                    return False
            claimed.append((start, end))
            return True

        def _room() -> int:
            return self.max_per_text - len(out)

        for c in propose_swaps(text, _DIRECTION_LOOKUP, _DIRECTION_RE, "temporal", max_per_text=self.max_per_text, rng=rng):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(c.span.start, c.span.end):
                continue
            out.append(c)

        for m in _PREPOST_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            surface = m.group(0)
            stem = m.group(1).lower()
            span = Span(surface, m.start(), m.end())
            if not _claim(span.start, span.end):
                continue
            repl = _PREPOST_LOOKUP[stem] + "-"
            out.append(Candidate(kind="temporal", span=span, replacement=repl, trigger=f"{surface} -> {repl}"))

        for m in _ISO_DATE_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            span = Span(m.group(0), m.start(), m.end())
            for delta in (1, -1):
                if len(out) >= self.max_per_text:
                    break
                shifted = _clamp_date(y + delta, mo, d)
                rendered = f"{shifted.year:04d}-{shifted.month:02d}-{shifted.day:02d}"
                out.append(Candidate(kind="temporal", span=span, replacement=rendered, trigger=f"{m.group(0)} -> {rendered}"))

        for m in _DMY_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                date(y, mo, d)
            except ValueError:
                continue
            span = Span(m.group(0), m.start(), m.end())
            for delta in (1, -1):
                if len(out) >= self.max_per_text:
                    break
                shifted = _clamp_date(y + delta, mo, d)
                rendered = f"{shifted.day:02d}/{shifted.month:02d}/{shifted.year:04d}"
                out.append(Candidate(kind="temporal", span=span, replacement=rendered, trigger=f"{m.group(0)} -> {rendered}"))

        for m in _MONTH_YEAR_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            month_name, year_str = m.group(1), m.group(2)
            year = int(year_str)
            span = Span(m.group(0), m.start(), m.end())
            for delta in (1, -1):
                if len(out) >= self.max_per_text:
                    break
                rendered = f"{month_name} {year + delta}"
                out.append(Candidate(kind="temporal", span=span, replacement=rendered, trigger=f"{m.group(0)} -> {rendered}"))

        for m in _QUARTER_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            q, year = int(m.group(1)), int(m.group(2))
            span = Span(m.group(0), m.start(), m.end())
            for delta in (1, -1):
                if len(out) >= self.max_per_text:
                    break
                new_q = q + delta
                new_year = year
                if new_q > 4:
                    new_q = 1
                    new_year += 1
                elif new_q < 1:
                    new_q = 4
                    new_year -= 1
                rendered = f"Q{new_q} {new_year}"
                out.append(Candidate(kind="temporal", span=span, replacement=rendered, trigger=f"{m.group(0)} -> {rendered}"))

        for m in _YEAR_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            year = int(m.group(0))
            span = Span(m.group(0), m.start(), m.end())
            for delta in (1, -1):
                if len(out) >= self.max_per_text:
                    break
                rendered = str(year + delta)
                out.append(Candidate(kind="temporal", span=span, replacement=rendered, trigger=f"{m.group(0)} -> {rendered}"))

        return out[: self.max_per_text]
