# Manuscript and reproducibility package

**Hypothesis-State Computing Under Causal Shift** is the research manuscript for Q-Neuro v0.1.0.
It reports a synthetic causal-benchmark study of ordered hypothesis-state computation. It is not a
clinical study, a medical device evaluation, or evidence of diagnostic performance on patients.

## Deliverables

- [`qneuro.docx`](qneuro.docx): editable Word manuscript
- [`qneuro.pdf`](qneuro.pdf): rendered publication PDF
- [`main.tex`](main.tex): modular LaTeX master with one generated file per major section
- [`source/`](source/): canonical Markdown prose
- [`tables/`](tables/): synchronized JSON and LaTeX tables generated from registered results
- [`figures/`](figures/): synchronized PNG and vector-PDF figures with a SHA-256 manifest
- [`MANUSCRIPT_METADATA.json`](MANUSCRIPT_METADATA.json): counts, scope, and source checksums

The current package contains 13,291 source words, 20 data figures, nine generated tables, and 25
references. Word and PDF are 54 pages in the verified LibreOffice rendering.

## Build

From the repository root:

```bash
uv sync --extra dev --extra paper --frozen
make paper-source      # tables, LaTeX modules, and DOCX
make paper             # all sources plus an ignored verification PDF in paper/rendered/
make paper-release     # intentionally refresh the checked-in release PDF
make reproduce-paper   # tests, lint, dashboard, figures, and full manuscript
```

`make latex` compiles `main.tex` when `latexmk` is installed. The checked-in PDF is generated from
the Word manuscript with LibreOffice so the PDF and requested editable document share one layout.
LibreOffice can vary PDF timestamps and internal font-object order between visually identical
exports, so routine reproduction writes an ignored verification render; `make paper-release` is
the explicit operation that refreshes the audited release binary.

## Artifact chain

Registered JSON results under `experiments/results/QN-XXXXXX/` are the numeric source of truth.
Figure scripts read those artifacts and emit PNG/PDF pairs. `build_tables.py` emits JSON/LaTeX
tables. `build_manuscript.py` consumes the canonical prose, generated tables, figures, and
references to emit modular LaTeX and Word. LibreOffice renders the final PDF.

The paper preserves negative findings, superseded runs, limits on statistical power, and the
synthetic-only evidence boundary. AI assistance is disclosed in the manuscript; the human author
remains responsible for every claim and any future submission.
