from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_official_provenance_pins_expected_repositories_and_shas():
    data = yaml.safe_load((ROOT / "official_baselines/provenance.yaml").read_text())
    assert data["continual_backprop"]["repository"] == "shibhansh/loss-of-plasticity"
    assert data["continual_backprop"]["commit"] == "a6b79580d85f3025bdb601566d3627c5f489f13b"
    assert data["rigl"]["repository"] == "google-research/rigl"
    assert data["rigl"]["commit"] == "d39fc7d46505cb3196cb1edeb32ed0b6dd44c0f9"
    assert (ROOT / ".gitignore").read_text().find(".external/") >= 0
