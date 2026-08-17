"""Q-Neuro 3.0: task correctness, halting arithmetic, and the matched-inference-path invariant.

Lane B code, but the invariants below are what make the cycle-1 negative result trustworthy: if
the task's stated distance disagreed with the walk, or if Q4's inference path differed from Q3's,
the reliability comparison in `research/qneuro3/` would be measuring two different things.
"""

from __future__ import annotations

import torch

from qneuro3 import hardware
from qneuro3.elastic import (
    Q0Fixed,
    Q3Arrival,
    Q4Grounded,
    arrival_loss,
    occupied_nodes,
)
from qneuro3.tasks import chase_to_goal, chase_to_goal_weighted

NODES = 12
DEPTH = 6


def test_chase_to_goal_distance_agrees_with_the_walk() -> None:
    """The declared target must be the number of hops that actually lands on node 0."""

    batch = chase_to_goal(64, NODES, DEPTH, seed=7)
    walk = occupied_nodes(batch["perm"], batch["start"], NODES)
    for row, distance in enumerate(batch["target"].tolist()):
        assert int(walk[row, distance - 1]) == 0
        earlier = walk[row, : distance - 1]
        assert not bool((earlier == 0).any()), "the goal was reached before the declared distance"


def test_chase_to_goal_targets_are_within_range() -> None:
    batch = chase_to_goal(128, NODES, DEPTH, seed=11)
    assert int(batch["target"].min()) >= 1
    assert int(batch["target"].max()) <= DEPTH


def test_weighted_task_respects_its_difficulty_weights() -> None:
    """Zero weight on a distance must mean that distance never appears."""

    weights = [0, 0, 0, 1, 1, 1]
    batch = chase_to_goal_weighted(256, NODES, DEPTH, seed=3, weights=weights)
    assert int(batch["target"].min()) >= 4
    walk = occupied_nodes(batch["perm"], batch["start"], NODES)
    for row, distance in enumerate(batch["target"].tolist()):
        assert int(walk[row, distance - 1]) == 0


def test_permutation_is_a_single_cycle() -> None:
    """Every node reachable from any start; otherwise the distance is not well defined."""

    batch = chase_to_goal(8, NODES, DEPTH, seed=5)
    for row in range(batch["perm"].shape[0]):
        perm = batch["perm"][row]
        assert sorted(perm.tolist()) == list(range(NODES))
        seen, current = set(), 0
        for _ in range(NODES):
            current = int(perm[current])
            seen.add(current)
        assert len(seen) == NODES


def test_arrival_step_counts_are_in_range() -> None:
    batch = chase_to_goal(16, NODES, DEPTH, seed=13)
    torch.manual_seed(0)
    _, steps = Q3Arrival(NODES, 16, DEPTH)(batch["perm"], batch["start"])
    assert int(steps.min()) >= 1
    assert int(steps.max()) <= DEPTH


def test_first_arrival_distribution_is_normalised() -> None:
    """log_first must be a distribution over 'which step is the first to fire', up to truncation.

    The masses cannot exceed one; they fall short exactly by the probability of never firing.
    """

    batch = chase_to_goal(16, NODES, DEPTH, seed=17)
    torch.manual_seed(0)
    log_first, _ = Q3Arrival(NODES, 16, DEPTH)(batch["perm"], batch["start"])
    total = log_first.exp().sum(dim=1)
    assert bool((total <= 1.0 + 1e-6).all())
    assert bool((total > 0.0).all())


def test_arrival_loss_rewards_firing_at_the_true_distance() -> None:
    """Moving mass onto the correct step must lower the loss. Otherwise the objective is wrong."""

    distance = torch.tensor([3, 1, 6])
    wrong = torch.full((3, DEPTH), -3.0)
    right = wrong.clone()
    right[torch.arange(3), distance - 1] = -0.1
    assert float(arrival_loss(right, distance)) < float(arrival_loss(wrong, distance))


def test_q4_inference_path_is_identical_to_q3() -> None:
    """The reliability comparison is only meaningful if Q4 adds nothing at inference.

    Q4's auxiliary head exists solely to shape training. Loading the same weights into both models
    must give bit-identical halting, and Q4's extra parameters must all live in that head.
    """

    batch = chase_to_goal(16, NODES, DEPTH, seed=19)
    torch.manual_seed(0)
    q4 = Q4Grounded(NODES, 16, DEPTH)
    q3 = Q3Arrival(NODES, 16, DEPTH)
    q3.load_state_dict({k: v for k, v in q4.state_dict().items() if not k.startswith("where.")})

    log4, steps4 = q4(batch["perm"], batch["start"])
    log3, steps3 = q3(batch["perm"], batch["start"])
    assert torch.equal(steps4, steps3)
    assert torch.allclose(log4, log3)

    extra = sum(p.numel() for p in q4.parameters()) - sum(p.numel() for p in q3.parameters())
    assert extra == sum(p.numel() for p in q4.where.parameters())


def test_q4_positions_are_only_computed_when_asked() -> None:
    batch = chase_to_goal(8, NODES, DEPTH, seed=23)
    torch.manual_seed(0)
    q4 = Q4Grounded(NODES, 16, DEPTH)
    assert len(q4(batch["perm"], batch["start"])) == 2
    _, _, positions = q4(batch["perm"], batch["start"], with_positions=True)
    assert positions.shape == (8, DEPTH, NODES)


def test_fixed_depth_reports_its_depth_honestly() -> None:
    batch = chase_to_goal(8, NODES, DEPTH, seed=29)
    torch.manual_seed(0)
    _, steps = Q0Fixed(NODES, 16, DEPTH, DEPTH)(batch["perm"], batch["start"])
    assert bool((steps == float(DEPTH)).all())


def test_hardware_profile_is_self_consistent() -> None:
    """The budget must be a real fraction of what the machine reports, not a hard-coded constant."""

    profile = hardware.detect(quick=True)
    assert profile.total_memory_gib > 0
    assert 0 < profile.available_memory_gib <= profile.total_memory_gib
    assert profile.physical_cores >= 1
    assert profile.torch_threads >= 1
    budget = profile.memory_budget_bytes()
    assert 0 < budget < profile.total_memory_gib * 1024**3
    if not profile.mps_available:
        assert profile.device_for(10**9) == "cpu"


# --- Q-Neuro 3.0 final architecture -------------------------------------------------------------


class _CountingCore(torch.nn.Module):
    """A core whose feature at step k is a one-hot for k, so halting behaviour is exactly checkable."""

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.calls = 0

    def advance(self, state, context):
        self.calls += 1
        index = int(state[0, 0].item())
        feature = torch.zeros(state.shape[0], self.depth)
        feature[:, min(index, self.depth - 1)] = 1.0
        return state + 1.0, feature


def test_first_arrival_masses_never_exceed_one() -> None:
    from qneuro3.adaptive import first_arrival

    torch.manual_seed(0)
    p = torch.rand(32, 8)
    log_first, step = first_arrival(p)
    total = log_first.exp().sum(dim=1)
    assert bool((total <= 1.0 + 1e-6).all())
    assert int(step.min()) >= 1
    assert int(step.max()) <= 8


def test_first_arrival_shortfall_is_the_never_fires_probability() -> None:
    """The masses fall short by exactly prod(1 - p_k). That is the whole point of not renormalising."""

    from qneuro3.adaptive import first_arrival

    torch.manual_seed(1)
    p = torch.rand(16, 6).clamp(1e-6, 1 - 1e-6)
    log_first, _ = first_arrival(p)
    shortfall = 1.0 - log_first.exp().sum(dim=1)
    assert torch.allclose(shortfall, (1 - p).prod(dim=1), atol=1e-5)


def test_expected_max_halt_rises_towards_the_worst_case() -> None:
    """The ceiling on every per-example adaptive-compute method, stated as arithmetic."""

    from qneuro3.adaptive import expected_max_halt

    pmf = 0.8 ** torch.arange(1, 33, dtype=torch.double)
    values = [expected_max_halt(pmf, n) for n in (1, 8, 64, 1024)]
    assert values == sorted(values), "E[max] must be non-decreasing in batch size"
    assert abs(values[0] - 4.97) < 0.05
    assert values[-1] > 29.0
    assert values[-1] < 32.0


def test_expected_max_halt_at_batch_one_is_the_mean() -> None:
    from qneuro3.adaptive import expected_max_halt

    pmf = torch.tensor([0.5, 0.3, 0.2], dtype=torch.double)
    assert abs(expected_max_halt(pmf, 1) - (1 * 0.5 + 2 * 0.3 + 3 * 0.2)) < 1e-9


def test_plan_switches_off_early_exit_at_serving_batch() -> None:
    """The measured crossover, encoded. Above it, early exit is a penalty and must not be chosen."""

    from qneuro3.adaptive import plan

    pmf = 0.8 ** torch.arange(1, 33, dtype=torch.double)
    assert plan(pmf, 1).early_exit is True
    assert plan(pmf, 1).mode == "M2 Eco"
    assert plan(pmf, 256).early_exit is False
    assert plan(pmf, 256).mode == "M2 Throughput"
    assert plan(pmf, 1).predicted_speedup > plan(pmf, 64).predicted_speedup


def test_early_exit_actually_stops_early() -> None:
    """Without genuine termination the saving is nominal; this is the test that catches that."""

    from qneuro3.adaptive import PredicateHalting

    depth = 8
    core = _CountingCore(depth)
    model = PredicateHalting(core, depth, 4, depth)
    with torch.no_grad():
        model.halt.bias.fill_(0.0)
        model.halt.weight.zero_()
        model.halt.weight[0, 2] = 40.0  # fire hard at step 3

    core.calls = 0
    _, steps = model.infer(torch.zeros(5, 1), None, early_exit=True)
    assert int(steps[0]) == 3
    assert core.calls == 3, f"early exit ran {core.calls} steps, expected 3"

    core.calls = 0
    model.infer(torch.zeros(5, 1), None, early_exit=False)
    assert core.calls == depth


def test_halting_loss_prefers_firing_at_the_true_step() -> None:
    from qneuro3.adaptive import halting_loss

    true_step = torch.tensor([2, 5])
    answer = torch.tensor([1, 0])
    logits = torch.tensor([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    wrong = torch.full((2, 8), -3.0)
    right = wrong.clone()
    right[torch.arange(2), true_step - 1] = -0.05
    assert float(halting_loss(right, true_step, logits, answer)) < float(
        halting_loss(wrong, true_step, logits, answer)
    )


# --- execution policies ------------------------------------------------------------------------


class _ScriptedCore:
    """A core whose rows halt at prescribed depths, so runtime accounting is exactly checkable."""

    def __init__(self, depths: list[int], max_depth: int):
        self.depths = torch.tensor(depths)
        self.max_depth = max_depth

    def init_state(self, batch):
        return {"depth": batch["depth"].clone(), "tag": batch["tag"].clone()}

    def step(self, state, position):
        halt = (position + 1 >= state["depth"]).float()
        logits = torch.stack([state["tag"].float(), -state["tag"].float()], dim=-1)
        return state, halt, logits


def _scripted(depths):
    core = _ScriptedCore(depths, max_depth=max(depths))
    batch = {
        "depth": torch.tensor(depths),
        "tag": torch.arange(len(depths)),
    }
    return core, batch


def test_lockstep_pays_the_batch_maximum() -> None:
    from qneuro3.runtime import lockstep

    core, batch = _scripted([1, 2, 5])
    run = lockstep(core, batch)
    assert run.example_steps == 3 * 5, "lockstep must advance every row to the slowest one"
    assert run.steps.tolist() == [1.0, 2.0, 5.0]


def test_compaction_executes_the_sum_of_depths() -> None:
    """The whole point: `Σ d_i` rather than `n · max d_i`."""

    from qneuro3.runtime import compacted

    core, batch = _scripted([1, 2, 5])
    run = compacted(core, batch)
    assert run.example_steps == 1 + 2 + 5
    assert run.steps.tolist() == [1.0, 2.0, 5.0]


def test_every_policy_agrees_with_lockstep() -> None:
    """A runtime that changes the answer is a different model. This is the check, not a comment."""

    from qneuro3.runtime import bucketed, compacted, continuous, lockstep, verify_equivalence

    depths = [3, 1, 4, 1, 5, 2, 6, 2]
    core, batch = _scripted(depths)
    reference = lockstep(core, batch)
    for candidate in (
        compacted(core, batch),
        compacted(core, batch, every=3),
        bucketed(core, batch, torch.tensor(depths, dtype=torch.float), buckets=3),
        continuous(core, batch, width=3),
    ):
        verify_equivalence(reference, candidate)


def test_deferred_compaction_does_not_overwrite_an_answer() -> None:
    """With `every > 1` a fired row keeps advancing; without a fired mask it overwrites itself.

    This was a real bug, caught by the equivalence check rather than by inspection.
    """

    from qneuro3.runtime import compacted, lockstep, verify_equivalence

    core, batch = _scripted([1, 1, 1, 8])
    verify_equivalence(lockstep(core, batch), compacted(core, batch, every=4))


def test_verify_equivalence_actually_rejects() -> None:
    from qneuro3.runtime import Execution, verify_equivalence

    good = Execution(torch.zeros(2, 2), torch.tensor([1.0, 2.0]), 3, 2, 0)
    wrong_steps = Execution(torch.zeros(2, 2), torch.tensor([1.0, 3.0]), 3, 2, 0)
    wrong_answers = Execution(torch.ones(2, 2), torch.tensor([1.0, 2.0]), 3, 2, 0)
    for bad in (wrong_steps, wrong_answers):
        try:
            verify_equivalence(good, bad)
        except ValueError:
            continue
        raise AssertionError("verify_equivalence accepted a policy that changed the result")


def test_plan_chooses_compaction_only_when_the_step_cost_justifies_it() -> None:
    """The measured boundary, encoded: compaction pays on an expensive core, not on a cheap one."""

    from qneuro3.adaptive import plan

    pmf = 0.85 ** torch.arange(1, 25, dtype=torch.double)
    assert plan(pmf, 256, step_cost_us=2.66).policy == "compacted"
    assert plan(pmf, 256, step_cost_us=0.33).policy != "compacted"
    assert plan(pmf, 256).policy != "compacted", "no step cost means compaction is unjustified"
    assert plan(pmf, 1, step_cost_us=2.66).policy == "lockstep", "nothing to compact at batch 1"


# --- final-phase invariants ----------------------------------------------------------------------


def test_configure_refuses_impossible_constraints() -> None:
    """A configurator that relaxes a constraint to return something is worse than one that fails."""

    from qneuro3.adaptive import configure

    configure(batch=1)  # satisfiable
    for kwargs in ({"batch": 1, "max_latency_us": 10.0}, {"batch": 1, "max_bytes": 1}):
        try:
            configure(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"configure({kwargs}) should have refused")


def test_configure_never_trades_quality() -> None:
    from qneuro3.adaptive import configure

    try:
        configure(batch=1, min_quality=0.99, measured_quality=0.80)
    except ValueError:
        return
    raise AssertionError("configure returned a point below the requested quality floor")


def test_peak_memory_favours_compaction() -> None:
    """The second Pareto axis: compaction retains live rows, attribution retains every step."""

    from qneuro3.adaptive import peak_activation_bytes

    full = peak_activation_bytes(256, 24, 64, policy="select")
    compact = peak_activation_bytes(256, 24, 64, policy="compacted", mean_depth=6.14)
    assert compact < full
    assert full / compact > 3.0


def test_har_validation_subjects_are_fixed_in_advance() -> None:
    """The held-out subjects must be a constant, not something chosen after seeing results.

    The real-data comparison is only credible if the split could not be tuned; the four lowest
    subject IDs were fixed in the frozen protocol before any outcome was read.
    """

    from research.qneuro3.har import VALIDATION_SUBJECTS

    assert VALIDATION_SUBJECTS == (1, 3, 5, 6)
    assert list(VALIDATION_SUBJECTS) == sorted(VALIDATION_SUBJECTS)


def test_every_frozen_q3_prediction_verifies_from_disk() -> None:
    """A frozen prediction whose hash cannot be re-verified from disk is not frozen."""

    import hashlib
    import json
    import pathlib

    frozen = pathlib.Path("research/qneuro3")
    names = [p for p in sorted(frozen.glob("QNEURO3-*.json")) if not p.name.endswith("-RESULT.json")]
    checked = 0
    for path in names:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "prediction" not in payload:
            continue
        key = "sha256" if "sha256" in payload else "sha256_of_prediction"
        recomputed = hashlib.sha256(
            json.dumps(payload["prediction"], indent=2, sort_keys=True).encode()
        ).hexdigest()
        assert recomputed == payload[key], f"{path.name} hash does not round-trip"
        checked += 1
    assert checked >= 10, f"expected at least 10 frozen predictions, found {checked}"
