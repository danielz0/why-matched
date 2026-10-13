# Contributing

## Setup

```bash
git clone git@github.com:danielz0/why-matched.git
cd why-matched
python -m venv .venv && source .venv/bin/activate
make install
pip install pre-commit && pre-commit install
```

`make install` runs `pip install -e ".[local,viz,dev]"`. See the
[`Makefile`](Makefile) for every available target (`install`, `test`,
`test-fast`, `lint`, `bench`).

## Branch naming

Prefix branches with the workstream they belong to, e.g.:

- `ws-a/validation-harness`
- `ws-d/subword-align`

## Definition of done

A PR merges when all of these hold:

- [ ] `make test` passes locally and in CI (all jobs)
- [ ] New stochastic paths take `seed`, use a local RNG, and have a
      same-seed-same-output test
- [ ] New perturbation rules extend
      `tests/perturbations/test_disjointness.py`
- [ ] Any behavior change is in `CHANGELOG.md` under **Behavior changes**
- [ ] `tests/test_collapse_legacy_fixture.py` unmodified, or the PR body
      starts with `BEHAVIOR CHANGE:` and explains why
- [ ] Public API changes update `__all__` in `whymatched/__init__.py` and
      the README
- [ ] Anything touching `calibration.py` has a test using both
      `NegationBlindModel` **and** `NegationAwareModel` — a test proving
      you can detect collapse says nothing about your false-positive rate

That last item is the one to enforce hardest. It's the failure a reviewer
will probe first, and it's easy to skip because the tests still pass
without it.

## The frozen-fixture rule

`tests/test_collapse_legacy_fixture.py` opens with:

> Frozen fixture pinning `detect_collapse(..., legacy_rules=True)` to the
> exact 0.2.2 output on `tests/test_collapse.py`'s fixtures (that file is
> the gate and must not be edited). This test is the explicit "same seed ->
> same output" backward-compatibility guarantee the perturbation-engine
> refactor promises; if this test needs to change, the refactor broke
> something.

Concretely:

- **`tests/test_collapse.py` must not be edited.** It's the gate, not an
  ordinary test file.
- If an assertion in `tests/test_collapse_legacy_fixture.py` genuinely
  needs to change, the PR body must **start** with a `BEHAVIOR CHANGE:`
  line explaining why, and `CHANGELOG.md` must gain an entry under the
  `### Behavior changes` heading (see below) — the same convention already
  used for the `legacy_rules=False` fixes.

## Extending the perturbation-disjointness test

`tests/perturbations/test_disjointness.py` guards against a surface form
(e.g. `"not"`, `"required"`) triggering two different perturbation kinds on
the same occurrence. When you add a new perturbation kind:

1. Give its `Perturbation` class a `_lookup_keys()` classmethod that
   returns its registered surface forms (mirror
   `ComparativePerturbation`/`QuantifierPerturbation`/`ModalPerturbation`).
2. Add it to the `sets` dict inside `_rule_set_keys()` in
   `tests/perturbations/test_disjointness.py`.
3. If the new kind deliberately shares a surface form with an existing one
   (resolved via span-arbitration priority in `perturbations/engine.py`),
   add that form to the module's `_ALLOWED_OVERLAP` set with a short
   comment explaining why the overlap is intentional.

## CHANGELOG convention

`CHANGELOG.md` uses a running `## Unreleased` section. Behavior changes go
under the exact existing heading style:

```markdown
### Behavior changes (opt-in via `legacy_rules=False`; becomes the default
in the next minor version)
```

Reuse that heading verbatim (adjusting the opt-in clause only if it no
longer applies) rather than inventing a new one.

## Determinism requirement

Every new stochastic code path takes a `seed` parameter and uses a local
`random.Random(seed)` (or `numpy.random.default_rng(seed)`) instance —
never the global `random`/`numpy.random` state. This matches the existing
pattern throughout the codebase (`detect_collapse(..., seed: int = 0)`,
`fit_profile(..., seed: int = 0)`). Same-seed calls must produce
byte-identical output; add a test proving it.
