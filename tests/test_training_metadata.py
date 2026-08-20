import json
import os
import tempfile
from src.train import build_metadata, load_feature_schema, write_metadata


def test_metadata_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        schema = {
            "schema_version": "1.0.0",
            "features": [{"name": "Age", "data_type": "integer"}],
            "canonical_feature_order": ["Age"],
            "transformed_feature_names": ["Age"],
        }
        with open(os.path.join(tmpdir, "schema.json"), "w") as f:
            json.dump(schema, f)

        feature_schema = load_feature_schema(tmpdir)
        metadata = build_metadata(
            model_name="TestModel",
            metrics={"roc_auc": 0.8},
            feature_schema=feature_schema,
            timestamp="2025-01-01T00:00:00+00:00",
            model_path=os.path.join(tmpdir, "model.pkl"),
        )
        metadata_path = write_metadata(tmpdir, metadata)

        assert os.path.exists(metadata_path)
        with open(metadata_path, "r") as f:
            loaded = json.load(f)
        assert loaded["model_name"] == "TestModel"
        assert loaded["version"] == "1.0.0"
        assert "feature_schema" in loaded
        assert "metrics" in loaded
