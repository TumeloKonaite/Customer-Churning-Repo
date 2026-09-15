from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.mlops.training_artifacts import TrainingArtifactWriter


def test_failed_staging_preserves_the_previous_artifact_bundle(tmp_path, monkeypatch):
    output = tmp_path / "training"
    output.mkdir()
    marker = output / "publication-marker.txt"
    marker.write_text("previous-complete-bundle", encoding="utf-8")
    writer = TrainingArtifactWriter()
    monkeypatch.setattr(
        writer,
        "_write_bundle",
        Mock(side_effect=RuntimeError("staging failed")),
    )

    with pytest.raises(RuntimeError, match="staging failed"):
        writer.write(object(), cohorts=object(), config={}, output_dir=output)

    assert marker.read_text(encoding="utf-8") == "previous-complete-bundle"
    assert not list(tmp_path.glob(".training-staging-*"))
