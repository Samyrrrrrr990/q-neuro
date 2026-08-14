from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manuscript_inventory_is_complete() -> None:
    metadata = load_json(PAPER / "MANUSCRIPT_METADATA.json")
    manifest = load_json(PAPER / "figures" / "manifest.json")
    references = load_json(PAPER / "references.json")

    assert metadata["scope"] == "Synthetic research only; no clinical validation."
    assert metadata["source_word_count"] >= 12_000
    assert metadata["figure_count"] == manifest["count"] == 20
    assert metadata["table_count"] == 9
    assert metadata["reference_count"] == len(references) == 25

    for figure in manifest["figures"]:
        for suffix in ("png", "pdf"):
            artifact = PAPER / "figures" / f"{figure['name']}.{suffix}"
            assert artifact.stat().st_size > 1_000


def test_generated_sources_are_internally_linked() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    modules = re.findall(r"\\input\{([^}]+)\}", main)
    assert len(modules) == 16
    assert all((PAPER / f"{module}.tex").is_file() for module in modules)

    citations = set()
    for module in modules:
        text = (PAPER / f"{module}.tex").read_text(encoding="utf-8")
        for group in re.findall(r"\\citep\{([^}]+)\}", text):
            citations.update(group.split(","))
    bibliography = (PAPER / "references.bib").read_text(encoding="utf-8")
    bibliography_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert citations
    assert citations <= bibliography_keys


def test_generated_tables_and_delivery_files_exist() -> None:
    table_json = sorted((PAPER / "tables").glob("*.json"))
    table_tex = sorted((PAPER / "tables").glob("*.tex"))
    assert len(table_json) == len(table_tex) == 9
    assert all(load_json(path)["rows"] for path in table_json)

    assert (PAPER / "qneuro.docx").read_bytes().startswith(b"PK")
    assert (PAPER / "qneuro.pdf").read_bytes().startswith(b"%PDF-")
