"""Build the static dashboard payload only from versioned research artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ARTIFACT_SUFFIXES = {".json", ".md", ".yaml"}


def read_json(path: str) -> dict | list:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean(container: dict, metric: str) -> float:
    return round(float(container[metric]["mean"]), 6)


def markdown_rows(path: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] not in {"Claim", "Q-Neuro idea"}:
            rows.append(cells)
    return rows


def count_public_artifacts(directory: Path) -> int:
    """Count portable result files without machine-local checkpoints or logs."""
    return sum(
        path.is_file() and path.suffix.lower() in PUBLIC_ARTIFACT_SUFFIXES
        for path in directory.iterdir()
    )


def main() -> None:
    discovery = read_json("research/discovery/generated/candidate_registry.json")
    frontiers = read_json("research/discovery/generated/pareto_frontiers.json")
    surprises = read_json("research/discovery/generated/surprises.json")
    proposals = yaml.safe_load(
        (ROOT / "research/discovery/generated/proposals.yaml").read_text(encoding="utf-8")
    )
    sweep = read_json("experiments/results/QN-000008/metrics.json")
    robustness: list[dict] = []
    for model, summary in sweep["summary"].items():
        robustness.append(
            {
                "model": model,
                "in_domain": mean(summary["in_domain"], "top1"),
                **{
                    severity: mean(summary[severity]["across_worlds"], "top1")
                    for severity in ("nuisance", "mild", "moderate", "severe")
                },
            }
        )
    claims = [
        {
            "claim": row[0],
            "evidence": row[1],
            "counterevidence": row[2],
            "confidence": row[3],
            "status": row[4],
        }
        for row in markdown_rows("docs/CLAIMS.md")
        if len(row) == 5
    ]
    failures = re.findall(
        r"^## (.+)$",
        (ROOT / "docs/FAILED_IDEAS.md").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    experiments = []
    for directory in sorted((ROOT / "experiments/results").glob("QN-*")):
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiments.append(
            {
                "id": directory.name,
                "status": result.get("status", "unknown"),
                "artifact_count": count_public_artifacts(directory),
            }
        )
    payload = {
        "generated_from": [
            "QN-000008",
            "QN-000014",
            "QN-000016",
            "QN-000021",
            "QN-000023",
            "QN-000025",
            "QN-000026",
        ],
        "candidates": discovery,
        "frontiers": frontiers,
        "surprises": surprises,
        "proposals": proposals,
        "robustness": robustness,
        "claims": claims,
        "failures": failures,
        "experiments": experiments,
    }
    output = ROOT / "dashboard/data.js"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "window.QNEURO_DATA = " + json.dumps(payload, indent=2, sort_keys=True) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {output.relative_to(ROOT)} from {len(discovery)} candidates")


if __name__ == "__main__":
    main()
