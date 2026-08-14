# Manuscript and reproducibility package

**Exact Real Controls Overturn an Apparent Complex-Valued Robustness Advantage** is the Q-Neuro
v1.0.0 results article. It reports a preregistered falsification study on synthetic sequential task
families. It is not a clinical study, a medical-device evaluation, or evidence of quantum
computation in cognition.

## Deliverables

- [`qneuro.docx`](qneuro.docx): editable, accessibility-audited Word manuscript
- [`qneuro.pdf`](qneuro.pdf): verified 25-page publication PDF
- [`main.tex`](main.tex): modular LaTeX master with one generated file per major section
- [`source/`](source/): canonical Markdown prose
- [`figures/`](figures/): synchronized PNG and vector-PDF figure with a SHA-256 manifest
- [`MANUSCRIPT_METADATA.json`](MANUSCRIPT_METADATA.json): counts, scope, and source checksums

The release contains 6,957 source words, one four-panel central figure, no data tables, and 14
primary references. The Word and PDF versions share the same 25-page LibreOffice-verified layout.

## Build

From the repository root:

```bash
uv sync --extra dev --extra paper --frozen
make paper-source      # LaTeX modules, staged figure, and DOCX
make paper             # source plus an ignored verification PDF
make paper-release     # intentionally refresh the checked-in release PDF
make reproduce-paper   # tests, lint, dashboard, figures, and manuscript
```

`make latex` compiles `main.tex` when `latexmk` is installed. Routine reproduction writes an
ignored verification PDF; `make paper-release` is the explicit operation that refreshes the
audited release binary.

## Evidence boundary

Registered artifacts under `experiments/results/` are the numeric source of truth. The central
figure reads those artifacts directly, and `build_manuscript.py` consumes canonical prose,
references, and the generated figure. The paper preserves the historical positive result, the
exact-real falsification, the failed frozen law, every outcome-eligibility limit, and the unopened
QN-GRAND-001 benchmark.

The manuscript has not undergone peer review, independent replication, or external statistical
audit. AI coding and writing assistance is disclosed; the human author remains responsible for
every claim and any submission.
