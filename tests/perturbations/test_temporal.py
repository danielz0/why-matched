import random

import pytest

from whymatched.perturbations.temporal import TemporalPerturbation


@pytest.mark.parametrize(
    "text,expected_trigger",
    [
        ("revenue grew before the merger", "before -> after"),
        ("reported prior to the audit", "prior to -> following"),
        ("measured since last quarter", "since -> until"),
        ("pre-launch checks passed", "pre- -> post-"),
        ("the year 2020 was notable", "2020 -> 2021"),
    ],
)
def test_temporal_positive_cases(text, expected_trigger):
    p = TemporalPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert any(c.trigger == expected_trigger for c in candidates)


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "revenue grew steadily",
        "no dates or direction words at all",
    ],
)
def test_temporal_negative_cases(text):
    p = TemporalPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_feb_29_clamps_to_feb_28_instead_of_raising():
    p = TemporalPerturbation()
    candidates = p.propose("as of 2024-02-29 the policy changed", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "2024-02-29 -> 2025-02-28" in triggers
    assert "2024-02-29 -> 2023-02-28" in triggers


def test_iso_date_shift_is_zero_padded():
    p = TemporalPerturbation()
    candidates = p.propose("launched on 2024-01-31", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "2024-01-31 -> 2025-01-31" in triggers
    assert not any("2025-1-31" in t for t in triggers)


def test_dmy_date_shift():
    p = TemporalPerturbation()
    candidates = p.propose("filed 28/02/2024 today", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "28/02/2024 -> 28/02/2025" in triggers


def test_month_year_shift():
    p = TemporalPerturbation()
    candidates = p.propose("released in March 2024", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "March 2024 -> March 2025" in triggers


def test_quarter_shift_wraps_year():
    p = TemporalPerturbation()
    candidates = p.propose("reported in Q4 2024", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "Q4 2024 -> Q1 2025" in triggers
    assert "Q4 2024 -> Q3 2024" in triggers


def test_post_prefix_flip():
    p = TemporalPerturbation()
    candidates = p.propose("post-launch review scheduled", rng=random.Random(0))
    assert any(c.trigger == "post- -> pre-" for c in candidates)


def test_max_per_text_respected():
    text = "2001 2002 2003 2004 2005 2006 2007 2008 2009 2010"
    p = TemporalPerturbation(max_per_text=3)
    candidates = p.propose(text, rng=random.Random(0))
    assert len(candidates) == 3
