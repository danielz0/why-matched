# Governance

This describes how `whymatched` is actually run today, plus what changes
once a second person has commit rights. It's written to match reality, not
to project a process that doesn't exist yet.

## Today

- **Maintainer**: Daniel Zimnicki ([@danielz0](https://github.com/danielz0)),
  sole committer.
- **Releases**: informal and ad hoc. There are no git tags and no
  per-version `CHANGELOG.md` headers today — version bumps happen via a
  commit that updates `pyproject.toml`'s `version` and adds/amends the
  running `## Unreleased` section of `CHANGELOG.md`, then a manual `pip`
  build + PyPI publish. There is no fixed cadence.
- **Review**: none required — direct commits to `main`.
- **Decisions**: maintainer's call, informed by `CHANGELOG.md`'s
  `### Behavior changes` entries and the test suite (`make test`).

## Once a second committer exists

- PRs require review before merge to `main`.
- **Two reviewers** required for any change touching
  `src/whymatched/calibration.py`, `src/whymatched/perturbations/engine.py`
  (span arbitration), or a benchmark corpus schema — these are the places
  where a reasonable-looking change can silently invalidate published
  numbers (see [`CODEOWNERS`](CODEOWNERS)).
- Release cadence gets formalized (git tags, a real `CHANGELOG.md`
  per-version history) once there's more than one person whose work needs
  coordinating around a cut.
- `SECURITY.md`'s reporting contact should be revisited to include the new
  committer, not just the original maintainer.

## Adding a second committer

This is a GitHub repository-settings action (Settings → Collaborators, or
org team membership) — outside anything a coding session or PR can do.
It's listed here as the actual blocking step for everything above to start
applying, not as something already done.
