# Contributing to Morpheu

Thanks for helping! Issues and pull requests are welcome in **English or Portuguese**.

## Ways to contribute

- **Report a bug** or **request a feature** through the issue templates.
- **Share results**: listening notes comparing engines/emotions for a language are valuable — open a
  discussion or an issue with the `feedback` label.
- **Code**: pick an issue labelled `good first issue` or `help wanted`, or something from the roadmap in the
  README. For larger changes, open an issue first so we can agree on the approach.
- **Translations**: the UI is in Portuguese; an English (or other) locale is very welcome.

## Development setup

Follow the installation steps in the [README](README.md). The test suite does **not** need a GPU: it uses
fake engines and a fake speech recognizer.

```
cd backend && ..\.venv\Scripts\python.exe -m pytest -q
cd frontend && npm run build        # includes the TypeScript type check
```

## Workflow

- `main` holds stable versions only and is protected: every change goes through a pull request.
- Use one short-lived branch per change: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat: ...`, `fix: ...`, `docs: ...`).
- Keep pull requests focused, add or update tests, and say in the description **what you tested and what
  you could not test** (for example, changes verified only without a GPU).
- CI must pass (backend tests on Windows, frontend build).

## Code style

- Python 3.12, standard library first, type hints, small modules with a docstring that says *why*.
- Comments explain non-obvious decisions, not what the code already says.
- Match the surrounding code; no new dependency without a reason in the PR.
- Adding a TTS engine: implement `engines/base.py:Engine`, declare honest `ModelInfo` limits
  (`ref_min_sec`, `ref_max_sec`, `max_chunk_chars`, `uses_ref_text`), register it in `app.default_manager`,
  and pin its packages in `backend/requirements.lock.txt`.

## Third-party code and models

Only add code, data or model weights whose license allows redistribution (MIT, Apache-2.0, BSD…). Keep
vendored files under `backend/morpheu/vendor/<name>/` with their `LICENSE` and a `NOTICE` that states the
source and commit. Never add audio of people who did not consent.

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE) and that
you follow the [Code of Conduct](CODE_OF_CONDUCT.md).
