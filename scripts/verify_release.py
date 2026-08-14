"""Verify the frozen Q-Neuro publication package from cached, registered artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "manifest.json"
RELEASE_DATE = "2026-08-14"


def artifact_paths() -> list[Path]:
    fixed = [
        "README.md",
        "CITATION.cff",
        "RELEASE_NOTES.md",
        "RESULTS.md",
        "REPLICATION.md",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "docs/PREREGISTRATION_AMENDMENT_001.md",
        "docs/PREREGISTRATION_NEXT_PHASE.md",
        "experiments/results/QN-000028/metrics.json",
        "experiments/results/QN-000029/metrics.json",
        "experiments/results/QN-000031/metrics.json",
        "experiments/results/QN-000033/metrics.json",
        "experiments/results/QN-000040/metrics.json",
        "experiments/results/QN-000042/metrics.json",
        "experiments/results/QN-GRAND-001/decision.json",
        "experiments/results/QN-GRAND-001/preflight.json",
        "research/analyses/generated/falsification_phase.json",
        "research/claims.json",
        "research/failures.json",
        "research/laws/FROZEN_CANDIDATE_001.json",
        "research/review_report.json",
        "research/figures/generated/falsification_phase.pdf",
        "research/figures/generated/falsification_phase.png",
        "dashboard/data.js",
        "paper/MANUSCRIPT_METADATA.json",
        "paper/build_manuscript.py",
        "paper/figures/manifest.json",
        "paper/qneuro.docx",
        "paper/qneuro.pdf",
        "paper/references.json",
        "qneuro/__init__.py",
        "scripts/verify_release.py",
    ]
    sources = sorted((ROOT / "paper" / "source").glob("*.md"))
    return sorted([ROOT / item for item in fixed] + sources)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, check_id: str, passed: bool, evidence: Any) -> None:
        self.checks.append({"check_id": check_id, "passed": bool(passed), "evidence": evidence})

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def verify_semantics(recorder: Recorder) -> dict[str, Any]:
    discovery = load_json("experiments/results/QN-000040/metrics.json")
    heldout = load_json("experiments/results/QN-000042/metrics.json")
    synthesis = load_json("research/analyses/generated/falsification_phase.json")
    law = load_json("research/laws/FROZEN_CANDIDATE_001.json")
    grand = load_json("experiments/results/QN-GRAND-001/decision.json")
    review = load_json("research/review_report.json")
    metadata = load_json("paper/MANUSCRIPT_METADATA.json")

    discovery_effects = [float(row["difference"]) for row in discovery["paired_effects"]]
    recorder.add("discovery_effect_count", len(discovery_effects) == 2880, len(discovery_effects))
    recorder.add(
        "discovery_no_positive_effects",
        sum(value > 0 for value in discovery_effects) == 0,
        {"positive": sum(value > 0 for value in discovery_effects)},
    )
    discovery_mean = mean(discovery_effects)
    recorder.add(
        "discovery_mean_effect",
        math.isclose(discovery_mean, -0.03695312475028913, abs_tol=1e-14),
        discovery_mean,
    )
    recorder.add(
        "discovery_outcome_ineligible",
        discovery["outcome_eligible"] is False,
        discovery["outcome_eligible"],
    )

    heldout_effects = [float(row["difference"]) for row in heldout["paired_effects"]]
    recorder.add("heldout_effect_count", len(heldout_effects) == 1920, len(heldout_effects))
    recorder.add(
        "heldout_no_positive_effects",
        sum(value > 0 for value in heldout_effects) == 0,
        {
            "positive": sum(value > 0 for value in heldout_effects),
            "zero": sum(value == 0 for value in heldout_effects),
        },
    )
    heldout_mean = mean(heldout_effects)
    recorder.add(
        "heldout_mean_effect",
        math.isclose(heldout_mean, -0.009157985999869804, abs_tol=1e-14),
        heldout_mean,
    )
    recorder.add(
        "heldout_outcome_ineligible",
        heldout["outcome_eligible"] is False,
        heldout["outcome_eligible"],
    )

    winner_counts = Counter(row["best_real_model"] for row in heldout["paired_effects"])
    expected_winners = {"exact_real_block_operator": 1478, "real_polar_operator": 442}
    recorder.add(
        "heldout_best_real_winners", dict(winner_counts) == expected_winners, winner_counts
    )

    grouped: dict[tuple[Any, ...], dict[str, dict[str, float]]] = {}
    for row in heldout["records"]:
        key = (
            row["family"],
            row["train_size"],
            row["training_seed"],
            row["world_seed"],
            row["severity"],
        )
        grouped.setdefault(key, {})[row["model"]] = row["metrics"]
    differences = {"top1": [], "nll": [], "ece": []}
    for models in grouped.values():
        complex_metrics = models["complex_operator"]
        exact_metrics = models["exact_real_block_operator"]
        for metric, metric_differences in differences.items():
            metric_differences.append(abs(complex_metrics[metric] - exact_metrics[metric]))
    maxima = {metric: max(values) for metric, values in differences.items()}
    expected_maxima = synthesis["heldout_confirmation"]["exact_real_equivalence"]
    equivalence_ok = (
        maxima["top1"] == expected_maxima["maximum_absolute_top1_difference"]
        and maxima["nll"] == expected_maxima["maximum_absolute_nll_difference"]
        and maxima["ece"] == expected_maxima["maximum_absolute_ece_difference"]
    )
    recorder.add("exact_real_equivalence", equivalence_ok, maxima)

    bootstrap = synthesis["heldout_confirmation"]["hierarchical_bootstrap"]
    recorder.add(
        "hierarchical_interval_nonpositive",
        bootstrap["ci_low"] < bootstrap["ci_high"] < 0,
        {"low": bootstrap["ci_low"], "high": bootstrap["ci_high"]},
    )
    confirmation = heldout["law_confirmation"]
    recorder.add(
        "frozen_law_fails_magnitude",
        confirmation["all_thresholds_pass"] is False
        and confirmation["threshold_checks"]["r2_at_least_minimum"] is False
        and confirmation["threshold_checks"]["mae_at_most_maximum"] is False,
        {"r2": confirmation["r2"], "mae": confirmation["mean_absolute_error"]},
    )
    recorder.add(
        "law_prohibited_from_grand_use",
        law["scope"]["qn_grand_001_permitted"] is False
        and law["scope"]["outcome_e_permitted"] is False,
        law["scope"],
    )

    grand_ok = (
        grand["status"] == "blocked_before_execution"
        and grand["qn_grand_001_executed"] is False
        and grand["sealed_benchmark_opened"] is False
        and grand["primary_confirmatory_effect_estimated"] is False
        and len(grand["blocking_failures"]) == 6
    )
    recorder.add("grand_benchmark_sealed", grand_ok, grand)
    recorder.add(
        "adversarial_claim_review",
        review["status"] == "pass" and not review["errors"],
        review,
    )

    docx_path = ROOT / "paper" / "qneuro.docx"
    try:
        with ZipFile(docx_path) as archive:
            docx_ok = archive.testzip() is None and "word/document.xml" in archive.namelist()
    except BadZipFile:
        docx_ok = False
    recorder.add("docx_container_valid", docx_ok, str(docx_path.relative_to(ROOT)))
    recorder.add(
        "docx_matches_manuscript_metadata",
        sha256(docx_path) == metadata["docx_sha256"],
        metadata["docx_sha256"],
    )
    recorder.add(
        "manuscript_counts",
        metadata["source_word_count"] == 6957
        and metadata["figure_count"] == 1
        and metadata["table_count"] == 0
        and metadata["reference_count"] == 14,
        {
            key: metadata[key]
            for key in ("source_word_count", "figure_count", "table_count", "reference_count")
        },
    )
    pdf_path = ROOT / "paper" / "qneuro.pdf"
    pdf_ok = pdf_path.read_bytes()[:5] == b"%PDF-" and pdf_path.stat().st_size > 100_000
    recorder.add("pdf_container_present", pdf_ok, pdf_path.stat().st_size)

    dashboard = (ROOT / "dashboard" / "data.js").read_text(encoding="utf-8")
    dashboard_tokens = [
        "A_falsified_intrinsic_complex_advantage",
        "blocked_before_execution",
        '"exact_real_block_operator": 1478',
        '"r2": -30.936351721073724',
    ]
    recorder.add(
        "dashboard_contains_frozen_synthesis",
        all(token in dashboard for token in dashboard_tokens),
        dashboard_tokens,
    )

    return {
        "discovery_effects": len(discovery_effects),
        "heldout_effects": len(heldout_effects),
        "total_independent_task_effects": len(discovery_effects) + len(heldout_effects),
        "heldout_mean_complex_minus_best_real": heldout_mean,
        "heldout_hierarchical_ci": [bootstrap["ci_low"], bootstrap["ci_high"]],
        "exact_real_maximum_absolute_difference": maxima,
        "frozen_law_heldout_r2": confirmation["r2"],
        "frozen_law_heldout_mae": confirmation["mean_absolute_error"],
        "grand_status": grand["status"],
    }


def write_manifest(summary: dict[str, Any]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in artifact_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot write release manifest; missing artifacts: {missing}")
    entries = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in artifact_paths()
    ]
    payload = {
        "schema_version": "1.0.0",
        "release_date": RELEASE_DATE,
        "scope": "Cached-artifact verification of synthetic, nonclinical research outputs.",
        "semantic_summary": summary,
        "artifacts": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def verify_manifest(recorder: Recorder) -> None:
    if not MANIFEST_PATH.is_file():
        recorder.add("release_manifest_present", False, str(MANIFEST_PATH.relative_to(ROOT)))
        return
    manifest = load_json("release/manifest.json")
    entries = manifest.get("artifacts", [])
    expected_paths = [str(path.relative_to(ROOT)) for path in artifact_paths()]
    manifest_paths = [entry["path"] for entry in entries]
    recorder.add(
        "manifest_artifact_set",
        manifest_paths == expected_paths,
        {"expected": len(expected_paths), "recorded": len(manifest_paths)},
    )
    mismatches: list[dict[str, Any]] = []
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            mismatches.append(
                {
                    "path": entry["path"],
                    "expected_bytes": entry["bytes"],
                    "actual_bytes": actual_size,
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": actual_hash,
                }
            )
    recorder.add("manifest_hashes", not mismatches, mismatches)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Freeze current publication artifacts into release/manifest.json.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON verification report path.")
    args = parser.parse_args()

    recorder = Recorder()
    summary = verify_semantics(recorder)
    if args.write_manifest:
        if not recorder.passed:
            raise SystemExit("Semantic checks failed; refusing to write a release manifest.")
        write_manifest(summary)
    verify_manifest(recorder)

    report = {
        "schema_version": "1.0.0",
        "status": "pass" if recorder.passed else "fail",
        "scope": "cached registered artifacts; no model retraining or independent replication",
        "summary": summary,
        "checks": recorder.checks,
    }
    if args.report:
        report_path = args.report if args.report.is_absolute() else ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"release verification: {report['status']} "
        f"({sum(check['passed'] for check in recorder.checks)}/{len(recorder.checks)} checks)"
    )
    if not recorder.passed:
        for check in recorder.checks:
            if not check["passed"]:
                print(f"FAILED {check['check_id']}: {check['evidence']}")
    return 0 if recorder.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
