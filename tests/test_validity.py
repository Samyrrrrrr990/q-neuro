from neuroworld import NeuroWorld
from neuroworld.validity import audit_dataset, case_consistency_errors, duplicate_rate


def test_case_consistency_and_split_overlap_checks() -> None:
    world = NeuroWorld()
    train = world.generate(120, seed=101)
    validation = world.generate(60, seed=102)
    test = world.generate(60, seed=103)
    assert case_consistency_errors((*train, *validation, *test)) == []
    assert duplicate_rate(train, train) == 1.0
    assert duplicate_rate(train, test) == 0.0


def test_shortcut_audit_is_deterministic_and_complete() -> None:
    world = NeuroWorld()
    result = audit_dataset(
        world.generate(160, seed=201),
        world.generate(80, seed=202),
        world.generate(80, seed=203),
    )
    names = {value["name"] for value in result["shortcuts"]}
    assert names == {
        "class_prior",
        "metadata_only",
        "sequence_length_only",
        "positive_negative_count",
        "order_only",
        "edge_token_identity",
        "single_feature",
        "depth_two_lookup",
        "nearest_neighbor",
    }
    assert result["case_consistency_errors"] == []
    assert result["train_test_duplicate_rate"] == 0.0


def test_repaired_world_removes_label_demographics_and_shares_nuisance_stages() -> None:
    world = NeuroWorld(demographic_signal_strength=0.0, shared_nuisance_stages=True)
    assert set(world._age_means.tolist()) == {0.5}
    assert set(world._sex_probs.tolist()) == {0.5}
    assert (world._stages[8:] == world._stages[8]).all()
