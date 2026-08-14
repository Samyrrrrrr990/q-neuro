# Replication and verification

This release supports two different activities that must not be conflated.

1. **Cached-artifact verification** checks the registered result chain, exact-control calculations,
   frozen-law decision, sealed-benchmark status, publication files, dashboard payload, and SHA-256
   manifest. It does not retrain models.
2. **Experimental reproduction** reruns selected synthetic training programs from their frozen
   configs. It consumes substantially more compute, produces new immutable experiment IDs, and is
   still not an independent replication unless performed and audited by an independent team.

The repository contains no patient data and no clinical replication pathway.

## Fresh-clone cached verification

```bash
git clone https://github.com/Samyrrrrrr990/q-neuro.git
cd q-neuro
git switch codex/q-neuro-falsification
uv sync --extra dev --extra paper --frozen
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/build_dashboard_data.py
uv run python paper/build_manuscript.py
uv run python scripts/verify_release.py
```

The manuscript build intentionally refreshes `paper/qneuro.docx`, modular LaTeX, and staged figure
files. It does not refresh the checked-in `paper/qneuro.pdf`; PDF exports can differ in timestamps
and internal object order across LibreOffice versions. To visually verify the PDF layout, install
LibreOffice and Poppler, render the rebuilt DOCX into a temporary directory, and compare page
images rather than binary hashes.

The expected cached-verification result is a passing test suite and:

```text
release verification: pass
```

`release/manifest.json` records the byte size and SHA-256 digest of every frozen publication
artifact. A mismatch is a failure until the scientific reason for the change is reviewed and a new
release manifest is intentionally generated with:

```bash
uv run python scripts/verify_release.py --write-manifest \
  --report release/verification_report.json
```

## Recompute the derived synthesis

These commands recompute the public synthesis, central figure, claim audit, dashboard payload, and
manuscript from already registered experiment outputs:

```bash
uv run python -m research.analyses.analyze_falsification_phase
uv run python -m research.figures.generate_falsification_phase
uv run python -m research.adversarial_reviewer
uv run python scripts/build_dashboard_data.py
uv run --extra paper python paper/build_manuscript.py
uv run python scripts/verify_release.py
```

## Rerun the next-phase experiments

Run smoke profiles before any compute-intensive profile. Every runner writes a new immutable
result directory and must never overwrite the release evidence.

```bash
# Leakage and task-surface checks
make simulator-red-team
make shift-gauntlet

# Outcome-ineligible pilot and mechanism work
make smoke-shift-pilot
make shift-pilot
make smoke-mechanism-suite
make mechanism-suite
make computational-law-suite

# Independent task-family discovery and untouched confirmation
make independent-task-audit
make smoke-independent-discovery
make independent-discovery
make freeze-candidate-law
make smoke-independent-confirmation
make independent-confirmation
```

Do not run `make qn-grand-001` as a way to “complete” the paper. The registered preflight has
already blocked QN-GRAND-001 before execution and the sealed benchmark remains unopened. A future
grand study requires a new protocol and new authorization after all six missing gates are actually
satisfied; it may not relabel the reduced studies as confirmatory.

## Expected scientific invariants

- QN-000040 contains 2,880 paired effects, zero positive, with mean −0.0369531.
- QN-000042 contains 1,920 paired effects, zero positive, with mean −0.00915799.
- Complex and exact-real top-1 match in all 1,920 held-out cells.
- QN-LAW-001 remains frozen and fails held-out magnitude thresholds: R² −30.936 and MAE 0.031265.
- QN-GRAND-001 remains `blocked_before_execution`, `executed=false`, and
  `sealed_benchmark_opened=false`.
- `research/review_report.json` remains `status=pass` with no errors.

## Replication status

The release has internal repetition across pilot, discovery, and held-out generator families plus a
fresh-clone cached-artifact rehearsal. It has **not** been independently replicated, peer reviewed,
clinically validated, or externally audited. Those are future scientific activities, not properties
that can be inferred from a passing software pipeline.
