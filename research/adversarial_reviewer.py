"""Fail the release when the machine-readable claim ledger outruns immutable evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CLAIM_FIELDS = {
    "claim_id",
    "exact_wording",
    "evidence",
    "counterevidence",
    "supporting_experiments",
    "effect_size",
    "uncertainty",
    "assumptions",
    "scope",
    "external_validity",
    "novelty_status",
    "replication_status",
    "falsifier",
    "confidence",
}
PROHIBITED_POSITIVE_PHRASES = (
    "clinically validated",
    "quantum brain",
    "quantum mechanism of cognition",
    "universally superior",
    "nobel prize",
    "most powerful ai",
)


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    claims = _load("research/claims.json")["claims"]
    failures = _load("research/failures.json")["failures"]
    synthesis = _load("research/analyses/generated/falsification_phase.json")
    grand = _load("experiments/results/QN-GRAND-001/decision.json")
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        missing = REQUIRED_CLAIM_FIELDS - set(claim)
        if missing:
            errors.append(f"{claim.get('claim_id', '<unknown>')} missing {sorted(missing)}")
        claim_id = str(claim.get("claim_id"))
        if claim_id in seen:
            errors.append(f"duplicate claim ID: {claim_id}")
        seen.add(claim_id)
        wording = str(claim.get("exact_wording", "")).lower()
        for phrase in PROHIBITED_POSITIVE_PHRASES:
            if phrase in wording and not wording.startswith("the repository contains no evidence"):
                errors.append(f"{claim_id} contains prohibited phrase: {phrase}")
        for experiment_id in claim.get("supporting_experiments", []):
            if not (ROOT / "experiments" / "results" / experiment_id).exists():
                errors.append(f"{claim_id} cites missing experiment: {experiment_id}")
    if synthesis["outcome_category"] != "A_falsified_intrinsic_complex_advantage":
        errors.append("synthesis does not preserve the falsified intrinsic-advantage decision")
    if grand["qn_grand_001_executed"] or grand["sealed_benchmark_opened"]:
        errors.append("grand-test ledger unexpectedly reports sealed outcome access")
    if not any(item["experiment_id"] == "QN-GRAND-001" for item in failures):
        errors.append("failure ledger omits the grand preflight block")
    if len(claims) < 5:
        warnings.append("claim ledger is unusually small")
    return {
        "status": "pass" if not errors else "fail",
        "claims_checked": len(claims),
        "failures_checked": len(failures),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    report = audit()
    output = ROOT / "research" / "review_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
