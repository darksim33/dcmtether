# DCMtether

Lossless DICOM ⇄ NIfTI round trips for MRI series, as a basis for derivatives (segmentations,
ADC/QSM maps) that are written back as properly referenced DICOM (SEG, Parametric Map).

**Status:** early development (Phase 0 of `docs/dev/PLAN.md`). The CLI commands exist but are
not implemented yet. **Research use only — not for clinical or diagnostic use.**

## Development

```
uv sync
uv run dcmtether --help
uv run ruff format --check . && uv run ruff check . && uv run pytest
```

The first test run downloads public test data and the DICOM standard into `.cache/`.
See `CLAUDE.md` for conventions and data-protection rules, and `docs/dev/test-data.md` for
what the test data covers.
