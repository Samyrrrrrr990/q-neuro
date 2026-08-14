# Contributing

Q-Neuro welcomes reproducibility fixes, stronger controls, independently designed generators, and
well-powered falsification studies.

1. Open an issue describing the hypothesis, control, primary metric, and falsifier.
2. Add or update a configuration before inspecting test outcomes.
3. Preserve completed result directories and register failures or superseded runs.
4. Run `make lint test`; regenerate analyses, figures, and dashboard when artifacts change.
5. Update `RESULTS.md`, `RESEARCH_LOG.md`, `docs/CLAIMS.md`, and `docs/FAILED_IDEAS.md` together.

Do not submit patient data, credentials, generated clinical advice, unsupported novelty language,
or result files whose provenance cannot be reproduced.
