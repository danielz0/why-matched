"""Numeric perturbation: scale currency/percentage/version/unit-bearing/bare
numbers by an order of magnitude (or transpose digits) so a magnitude-blind
embedding model can be caught. All arithmetic goes through Decimal -- never
float -- because division is not exact in float: 1.15/10 ==
0.11499999999999999, but Decimal("1.15")/10 == Decimal("0.115") exactly.
Since every scale operation here is a power of 10, the arithmetic is always
exact in base-10 Decimal; `_reformat` never rounds, it only chooses how many
decimal places to *display* (at least as many as the original had, and at
least as many as the exact result needs).
"""
from __future__ import annotations

import random
import re
from decimal import Decimal
from typing import List, Optional, Tuple

from ..text import Span
from .base import Candidate

_CURRENCY_RE = re.compile(r"[$€£]\s?[\d,]+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+){0,2}\b|\b\d+\.\d+\.\d+\b")
_UNITS = ("mg", "kg", "g", "GB", "MB", "KB", "ms", "km", "m", "lb", "oz")
_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s?(?:" + "|".join(_UNITS) + r")\b")
_BARE_NUM_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _reformat(original: str, new_value: Decimal) -> str:
    """Preserve thousands separators, decimal places (at minimum), leading
    zeros, and sign/style of `original` while rendering the exact
    `new_value`."""
    has_comma = "," in original
    original_int = original.split(".", 1)[0].replace(",", "").lstrip("-")
    original_decimals = len(original.split(".", 1)[1]) if "." in original else 0

    exact = new_value.normalize()
    _, _, exponent = exact.as_tuple()
    needed_decimals = -exponent if exponent < 0 else 0
    decimals = max(original_decimals, needed_decimals)

    neg = exact < 0
    exact = abs(exact)
    q_str = format(exact, "f")
    if "." in q_str:
        int_part_str, frac_part_str = q_str.split(".", 1)
    else:
        int_part_str, frac_part_str = q_str, ""

    if not has_comma and original_int.startswith("0") and len(original_int) > len(int_part_str):
        int_part_str = int_part_str.rjust(len(original_int), "0")

    if has_comma:
        rev = int_part_str[::-1]
        grouped = ",".join(rev[i : i + 3] for i in range(0, len(rev), 3))
        int_part_str = grouped[::-1]

    out = ("-" if neg else "") + int_part_str
    if decimals:
        out += "." + frac_part_str.ljust(decimals, "0")[:decimals]
    return out


def _extract_digits(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _scale_candidates(
    span: Span, kind: str, digits_text: str, prefix: str, suffix: str, *, include_divide: bool = True
) -> List[Candidate]:
    value = _extract_digits(digits_text)
    out = []
    times = value * 10
    if times != value:
        rendered = _reformat(digits_text, times)
        out.append(
            Candidate(kind=kind, span=span, replacement=prefix + rendered + suffix, trigger=f"{span.text} -> {prefix}{rendered}{suffix}")
        )
    if include_divide:
        div = value / 10
        if div != value:
            rendered = _reformat(digits_text, div)
            out.append(
                Candidate(kind=kind, span=span, replacement=prefix + rendered + suffix, trigger=f"{span.text} -> {prefix}{rendered}{suffix}")
            )
    return out


def _transpose_digits(digits: str) -> Optional[str]:
    """Swap the last two integer digits, skipping the decimal point.
    Returns None if fewer than 2 integer digits or the swap is a no-op."""
    int_part = digits.split(".", 1)[0].replace(",", "")
    frac = digits[len(digits.split(".", 1)[0]):] if "." in digits else ""
    if len(int_part) < 2:
        return None
    swapped = int_part[:-2] + int_part[-1] + int_part[-2]
    if swapped == int_part or swapped[0] == "0":
        return None
    return swapped + frac


class NumericPerturbation:
    name = "numeric"
    kind = "numeric"
    priority = 60
    requires: Tuple[str, ...] = ()

    def __init__(self, max_per_text: int = 8):
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        del rng
        out: List[Candidate] = []
        claimed: List[Tuple[int, int]] = []

        def _claim(start: int, end: int) -> bool:
            for s, e in claimed:
                if start < e and s < end:
                    return False
            claimed.append((start, end))
            return True

        def _budget() -> int:
            return self.max_per_text - len(out)

        for m in _CURRENCY_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            sym = m.group(0)[0]
            digits_text = m.group(0)[1:].strip()
            span = Span(m.group(0), m.start(), m.end())
            out.extend(_scale_candidates(span, "numeric", digits_text, sym, "")[: _budget()])

        for m in _PERCENT_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            raw = m.group(0)
            digits_text = raw.rstrip("% ").rstrip("%")
            suffix = raw[len(digits_text):]
            span = Span(raw, m.start(), m.end())
            out.extend(
                _scale_candidates(span, "numeric", digits_text, "", suffix, include_divide=False)[: _budget()]
            )
            transposed = _transpose_digits(digits_text)
            if transposed and len(out) < self.max_per_text:
                out.append(
                    Candidate(kind="numeric", span=span, replacement=transposed + suffix, trigger=f"{raw} -> {transposed}{suffix}")
                )

        for m in _VERSION_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            raw = m.group(0)
            prefix = "v" if raw.startswith("v") else ""
            parts = raw[len(prefix):].split(".")
            span = Span(raw, m.start(), m.end())
            bump_indices = [len(parts) - 1]
            if len(parts) > 1:
                bump_indices.append(len(parts) - 2)
            if len(parts) > 2:
                bump_indices.append(0)
            elif 0 not in bump_indices:
                bump_indices.append(0)
            seen_bumps = set()
            for bump_idx in bump_indices:
                if len(out) >= self.max_per_text:
                    break
                new_parts = list(parts)
                new_parts[bump_idx] = str(int(new_parts[bump_idx]) + 1)
                bumped = prefix + ".".join(new_parts)
                if bumped != raw and bumped not in seen_bumps:
                    seen_bumps.add(bumped)
                    out.append(Candidate(kind="numeric", span=span, replacement=bumped, trigger=f"{raw} -> {bumped}"))

        for m in _UNIT_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            raw = m.group(0)
            unit_match = re.search(r"[A-Za-z]+$", raw)
            unit = unit_match.group(0)
            digits_text = raw[: unit_match.start()].strip()
            sep = raw[len(digits_text) : unit_match.start()]
            span = Span(raw, m.start(), m.end())
            out.extend(_scale_candidates(span, "numeric", digits_text, "", sep + unit)[: _budget()])

        for m in _BARE_NUM_RE.finditer(text):
            if len(out) >= self.max_per_text:
                return out
            if not _claim(m.start(), m.end()):
                continue
            raw = m.group(0)
            span = Span(raw, m.start(), m.end())
            out.extend(_scale_candidates(span, "numeric", raw, "", "")[: _budget()])
            transposed = _transpose_digits(raw)
            if transposed and len(out) < self.max_per_text:
                out.append(Candidate(kind="numeric", span=span, replacement=transposed, trigger=f"{raw} -> {transposed}"))

        return out[: self.max_per_text]
