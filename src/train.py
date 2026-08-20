import argparse
import json
import os
from datetime import datetime, timezone

from src.pipeline.training_pipeline import TrainingPipeline


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def load_feature_schema(artifacts_dir):
    schema_path = os.path.join(artifacts_dir, "schema.json")
    schema = _load_json(schema_path, {})

    return {
        "schema_version": schema.get("schema_version", "legacy"),
        "raw_features": schema.get("features", schema.get("feature_schema", [])),
        "input_columns": schema.get(
            "canonical_feature_order", schema.get("all_cols", [])
        ),
        "transformed_columns": schema.get("transformed_feature_names", []),
    }


def build_metadata(model_name, metrics, feature_schema, timestamp, model_path):
    return {
        "trained_at": timestamp,
        "model_name": model_name,
        "version": feature_schema.get("schema_version", "1.0.0"),
        "model_path": model_path,
        "metrics": metrics,
        "feature_schema": feature_schema,
    }


def write_metadata(artifacts_dir, metadata):
    os.makedirs(artifacts_dir, exist_ok=True)
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Train churn model and write metadata artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        default=ARTIFACTS_DIR,
        help="Directory for training artifacts and metadata.json",
    )
    args = parser.parse_args()

    training_result = TrainingPipeline().run()
    model_name = training_result.get("best_model_name", "unknown-model")
    metrics = training_result.get("metrics", {})
    model_path = training_result.get("model_path", "")

    feature_schema = load_feature_schema(args.artifacts_dir)
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = build_metadata(
        model_name=model_name,
        metrics=metrics,
        feature_schema=feature_schema,
        timestamp=timestamp,
        model_path=model_path,
    )

    metadata_path = write_metadata(args.artifacts_dir, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
