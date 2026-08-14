# Clinical validation roadmap

Current status: **preclinical computational research only; not a medical device; not for patient
care.** No patient data, clinician study, prospective deployment, or clinical-benefit estimate is
present in this repository.

This roadmap is a boundary document, not a claim that Q-Neuro should enter clinical use. The
intrinsic complex-arithmetic hypothesis was falsified within the tested synthetic scope, and the
grand confirmation did not execute. Any clinical program would therefore need a new clinical
question and should compare the simplest surviving structured-real mechanism, not assume that a
complex implementation is preferred.

## Stage 0 — independent computational replication

- Independent team reproduces exact-real equivalence and key negative effects from a tagged
  release on separately controlled hardware.
- Complete comparator/search/compute requirements that blocked QN-GRAND-001.
- Decide whether any architecture claim remains worth clinical investigation.

Exit criterion: reproducible computational result with complete audit trail. Failure ends the
clinical path.

## Stage 1 — problem and governance definition

- Select one clinically meaningful decision, users, setting, and time horizon with clinicians and
  patients; avoid a generic “diagnosis AI” objective.
- Establish lawful data access, institutional review, privacy threat model, data minimization,
  security controls, incident response, and accountable clinical ownership.
- Specify intended use, contraindications, human oversight, and the harm of false positive, false
  negative, delayed, or overconfident output.

Exit criterion: approved protocol, data-management plan, and prospective statistical analysis
plan before test labels are accessed.

## Stage 2 — retrospective external validation

- Use multiple sites and time periods, with a locked patient-level split and deduplication.
- Compare against standard-of-care scores, simple statistical models, strong sequential baselines,
  and clinician judgment where appropriate.
- Measure discrimination, calibration, decision-curve utility, subgroup performance, missingness,
  prevalence shift, dataset drift, and failure severity.
- Evaluate abstention and human-readable uncertainty; top-1 accuracy alone is insufficient.

Exit criterion: clinically meaningful, calibrated benefit across sites with no unacceptable
subgroup harm. A non-inferior real model displaces a complex implementation.

## Stage 3 — prospective silent study

- Lock software and run without influencing care.
- Measure real-time data availability, latency, alert burden, workflow fit, drift, and silent safety
  events under prospective case accrual.
- Audit all exclusions and unavailable predictions.

Exit criterion: operational reliability and prospective performance consistent with the locked
analysis plan.

## Stage 4 — interventional evaluation

- Use an appropriately powered controlled study with independent monitoring.
- Evaluate patient- or process-relevant outcomes, not only model metrics.
- Predefine stopping for harm, subgroup disparity, automation bias, and workflow overload.

Exit criterion: demonstrated net benefit under human oversight. Retrospective accuracy cannot
substitute for this stage.

## Stage 5 — regulatory, deployment, and surveillance

- Determine jurisdiction-specific medical-device obligations with qualified regulatory counsel.
- Maintain version control, cybersecurity, usability engineering, risk management, change
  control, post-market monitoring, and rollback.
- Revalidate after material changes in data, workflow, model, or intended use.

No stage may be skipped because a synthetic benchmark result is strong. The present release meets
none of the clinical exit criteria.
