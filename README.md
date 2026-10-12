# DCMtether

Lossless DICOM ⇄ NIfTI round trips for MRI series, as a basis for derivatives (segmentations,
ADC/QSM maps) that are written back as properly referenced DICOM (SEG, Parametric Map, or
derived MR images for any PACS).

**Research use only — not for clinical or diagnostic use.** A converted dataset **identifies
patients** (file names are built from PatientName and StudyDate): never share it as is.

```
uv sync
uv run dcmtether run /path/to/dicom -o /path/to/dataset  # scan + convert (rerunnable; default output: the DICOM folder)
uv run dcmtether to-dcm DATASET/sub-…/….nii.gz rebuilt/  # back to identical DICOM
uv run dcmtether to-seg NIFTI labels.nii.gz seg-template.json seg.dcm
```

- [Usage](docs/usage.md): workflow and every command.
- [Formats](docs/formats.md): dataset layout, file names, `*.tether.json`, sidecar, manifest, log.
- [Supported inputs](docs/supported-inputs.md): what converts, what does not, and why.
- Example metadata templates: [`docs/examples/`](docs/examples/).

## Development

```
uv run ruff format --check . && uv run ruff check . && uv run pytest
```

The first test run downloads public test data and the DICOM standard into `.cache/`. Plan and
decisions: `docs/dev/PLAN.md`; conventions and data-protection rules: `CLAUDE.md`; test data
coverage: `docs/dev/test-data.md`.
