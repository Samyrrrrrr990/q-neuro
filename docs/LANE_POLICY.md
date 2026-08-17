# Lane policy: discovery and confirmation are physically separate

Status: active operating policy

Date: 2026-08-15

Governs: all work under `docs/ML2_PREREGISTRATION_001.md`

---

## Why this exists

Q-Neuro 1.0 failed in a specific way that is worth naming: research decisions repeatedly observed
similar held-out environments, so the same evidence informed both what to look for and whether it was
there. `docs/NEXT_PHASE_AUDIT.md` §3.6 called this development-set reuse at the research-program
level. No individual run cheated. The program still adapted itself onto its own test surface.

The only durable fix is structural. Discovery and confirmation must not share evidence, and the
boundary must be enforced by where files live and what the code refuses to do, not by intention.

---

## Lane A — confirmatory

**Contains:** preregistered hypotheses, frozen estimators, sealed families, frozen model
definitions, untouched seeds, preregistered statistics.

**Location:** `experiments/run_qe_*.py`, `experiments/results/QE-*`.

**Rules:**

- No modification after outcomes are observed. A correction needs a versioned amendment and a new
  experiment ID.
- One attempt per sealed family.
- Publication-eligible.
- A Lane A experiment may read a frozen estimator. It may never fit one.

## Lane B — discovery

**Contains:** exploratory sweeps, mechanism search, anomaly detection, phase-transition search,
invariant search, symbolic investigation, high-risk ideas.

**Location:** `research/discovery_lab/`, artifacts under `research/discovery_lab/generated/`.

**Rules:**

- Free to iterate, refit, and change direction.
- **Nothing in Lane B is a claim.** No Lane B artifact may be cited as evidence for a registered
  claim in `research/claims.json`.
- Every candidate gets a `DISCOVERY-NNN` identifier and the full record in §3 below.
- Lane B may not touch any family marked sealed in preregistration §5.1.

---

## The promotion path

A Lane B observation becomes a Lane A claim only by completing every step, in order:

1. **Exploration.** Observed in the discovery lab, with an effect size.
2. **Internal replication.** Reproduces across seeds and configurations without refitting.
3. **Mechanistic explanation.** A stated mechanism, and the *simplest boring explanation* tested
   first and either ruled out or accepted as the answer.
4. **Frozen hypothesis.** A prediction with numeric thresholds, serialized and hashed before any
   new evidence is generated.
5. **Untouched confirmatory test.** Evaluated once, in Lane A, on evidence that did not exist when
   the hypothesis was frozen.

**Never fit and validate on the same evidence.** A candidate that skips a step is not promoted; it
stays in Lane B with its status recorded.

---

## Discovery record schema

Every `DISCOVERY-NNN` records, at minimum:

| Field | Meaning |
|---|---|
| `observation` | What was seen, concretely |
| `effect` | Effect size and units |
| `scope` | Where it holds; where it was not tested |
| `competing_explanations` | Alternatives considered |
| `simplest_boring_explanation` | The dullest account that would also produce this, and whether it survives |
| `prior_art` | What already exists; whether the mechanism is known |
| `proposed_mechanism` | Why it should happen |
| `falsifier` | What observation would kill it |
| `replication_status` | Seeds, configurations, independent implementations |
| `promotion_stage` | Which of the five steps above is complete |
| `preregistration_eligible` | Whether it is ready to freeze |

Ranking is by effect size, reproducibility, cross-family occurrence, simplicity, predictive power,
novelty against prior art, and falsifiability — **not** primarily by p-value.

---

## What this policy forbids

- Reporting a Lane B result as a finding.
- Moving a family between splits after any outcome in it is observed.
- Adding features to an estimator after seeing how it scored.
- Re-running a failed idea under a new name until it looks successful. Failed ideas keep their
  original identifier in `research/failures.json` and `docs/FAILED_IDEAS.md`.
