from research.adversarial_reviewer import audit


def test_release_claims_pass_adversarial_audit() -> None:
    report = audit()
    assert report["status"] == "pass"
    assert report["errors"] == []
