"""Nova: task validity, model invariants, and the properties every comparison depends on.

Nova's conclusion is negative, so these tests exist to make the negative trustworthy. If the tasks
were solvable by a degenerate predictor, or a model were non-causal, or the parameter matching were
wrong, the whole capability frontier would be measuring something other than architecture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nova import tasks
from nova.candidates import CANDIDATES, InvariantAttention
from nova.zoo import BASELINES, match_parameters

CLEAN = ("parity_scan", "mod_sum", "copy", "reverse", "needle")
WEAK = ("cummax", "sort")


@pytest.mark.parametrize("name", list(tasks.TASKS))
@pytest.mark.parametrize("length", [8, 16, 32, 64])
def test_tasks_are_well_formed_at_every_length(name: str, length: int) -> None:
    batch = tasks.make(name, 8, length, seed=3)
    assert batch["x"].shape == (8, length)
    assert batch["y"].shape == (8, length)
    assert bool(batch["mask"].any()), "a task with no scored positions measures nothing"
    assert int(batch["x"].max()) < tasks.VOCAB
    assert int(batch["y"][batch["mask"]].max()) < tasks.VOCAB


def test_clean_tasks_resist_the_degenerate_predictors() -> None:
    """The five headline tasks must not be solvable from position alone."""

    for name in CLEAN:
        sample = tasks.make(name, 1024, 64, seed=11)
        best = 0.0
        for position in range(64):
            selected = sample["mask"][:, position]
            if bool(selected.any()):
                counts = torch.bincount(
                    sample["y"][:, position][selected], minlength=tasks.VOCAB
                )
                best = max(best, float(counts.max()) / float(selected.sum()))
        assert best < 0.60, f"{name} is solvable from position alone at {best:.3f}"


def test_the_dropped_tasks_really_were_weak() -> None:
    """cummax and sort were removed from headline scoring; this records why, executably."""

    for name in WEAK:
        sample = tasks.make(name, 1024, 64, seed=11)
        best = 0.0
        for position in range(64):
            selected = sample["mask"][:, position]
            if bool(selected.any()):
                counts = torch.bincount(
                    sample["y"][:, position][selected], minlength=tasks.VOCAB
                )
                best = max(best, float(counts.max()) / float(selected.sum()))
        assert best > 0.60, f"{name} was dropped as a weak instrument but scores only {best:.3f}"


def test_parity_and_mod_sum_targets_are_correct() -> None:
    batch = tasks.make("parity_scan", 32, 20, seed=5)
    bits = batch["x"] - tasks.FIRST
    assert torch.equal(batch["y"] - tasks.FIRST, bits.cumsum(1) % 2)
    batch = tasks.make("mod_sum", 32, 20, seed=5)
    values = batch["x"] - tasks.FIRST
    assert torch.equal(batch["y"] - tasks.FIRST, values.cumsum(1) % 7)


def test_needle_answer_is_the_value_of_the_queried_key() -> None:
    batch = tasks.make("needle", 64, 24, seed=6)
    query = batch["x"][:, -1]
    answer = batch["y"][:, -1]
    for row in range(64):
        sequence = batch["x"][row]
        # Keys sit at EVEN indices of the interleaved key/value region. Searching every position
        # can land on a VALUE that happens to equal the query, which is a property of the test
        # rather than of the task.
        found = None
        for index in range(0, len(sequence) - 2, 2):
            if int(sequence[index]) == int(query[row]) and sequence[index + 1] != tasks.BLANK:
                found = index
                break
        assert found is not None, "the queried key does not appear in the key positions"
        assert int(sequence[found + 1]) == int(answer[row])


@pytest.mark.parametrize("name", ["gru", "lstm", "transformer_rope", "ssm_diag", "causal_mlp"])
def test_baselines_are_causal_and_finite(name: str) -> None:
    config, _ = match_parameters(name, 120_000, depth=2 if name in ("gru", "lstm") else 3)
    torch.manual_seed(0)
    model = BASELINES[name](**config)
    x = tasks.make("parity_scan", 4, 16, seed=0)["x"]
    with torch.no_grad():
        first = model(x)
        altered = x.clone()
        altered[:, -1] = (altered[:, -1] + 1) % tasks.VOCAB
        second = model(altered)
    assert bool(torch.isfinite(first).all())
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5), "model is not causal"


def test_parameter_matching_is_actually_matched() -> None:
    """Comparing families at a fixed width would compare model sizes, not architectures."""

    counts = []
    for name in ("gru", "lstm", "transformer_rope", "ssm_diag"):
        _, count = match_parameters(name, 120_000, depth=2 if name in ("gru", "lstm") else 3)
        counts.append(count)
    assert max(counts) / min(counts) < 1.15


def test_max_normaliser_is_not_secretly_softmax() -> None:
    """The bug that invalidated the first run of the primary hypothesis, as a regression test.

    Dividing by the sum after dividing by the max is algebraically softmax. The two produced
    identical numbers and the hypothesis had not been tested at all.
    """

    torch.manual_seed(0)
    scores = torch.randn(1, 8, 32)
    soft = InvariantAttention(32, 1, "softmax", rope=False).eval()
    hard = InvariantAttention(32, 1, "max", rope=False).eval()
    hard.load_state_dict(soft.state_dict(), strict=False)
    with torch.no_grad():
        assert not torch.allclose(soft(scores), hard(scores), atol=1e-4)


def test_max_normaliser_is_more_length_invariant_than_softmax() -> None:
    """The operator-level property the hypothesis rested on. It holds; it just did not help."""

    def drift(normaliser: str) -> float:
        torch.manual_seed(0)
        attention = InvariantAttention(32, 1, normaliser, rope=False).eval()
        torch.manual_seed(2)
        query, key = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
        distractors = torch.randn(1, 24, 32) * 0.05
        with torch.no_grad():
            few = attention(torch.cat([key, query], 1))[0, -1]
            many = attention(torch.cat([key, distractors, query], 1))[0, -1]
        return float((few - many).abs().max())

    assert drift("max") < drift("softmax") * 0.6


@pytest.mark.parametrize("name", ["cursor", "cursor_attn", "rnn_attn_max", "late_fusion_gated"])
def test_candidates_are_causal_and_finite(name: str) -> None:
    BASELINES.setdefault(name, CANDIDATES[name])
    config, _ = match_parameters(name, 120_000, depth=2)
    torch.manual_seed(0)
    model = CANDIDATES[name](**config)
    x = tasks.make("copy", 4, 16, seed=0)["x"]
    with torch.no_grad():
        first = model(x)
        altered = x.clone()
        altered[:, -1] = 5
        second = model(altered)
    assert bool(torch.isfinite(first).all())
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5), f"{name} is not causal"


def test_every_frozen_nova_prediction_verifies_from_disk() -> None:
    frozen = Path("research/nova")
    names = [p for p in sorted(frozen.glob("NOVA-*.json")) if "RESULT" not in p.name]
    checked = 0
    for path in names:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "prediction" not in payload:
            continue
        recomputed = hashlib.sha256(
            json.dumps(payload["prediction"], indent=2, sort_keys=True).encode()
        ).hexdigest()
        assert recomputed == payload["sha256"], f"{path.name} hash does not round-trip"
        checked += 1
    assert checked >= 2
