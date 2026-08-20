import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.components.data_transformation import build_model_pipeline
from src.model_schema import (
    CANONICAL_FEATURE_ORDER,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    build_model_schema,
)
from src.pipeline import prediction_pipeline


def training_frame():
    return pd.DataFrame(
        [
            [600, "France", "Female", 30, 2, 0.0, 1, 1, 1, 50_000.0],
            [700, "Germany", "Male", 50, 8, 100_000.0, 2, 0, 0, 80_000.0],
            [650, None, "Female", np.nan, 4, 30_000.0, 1, 1, 0, 60_000.0],
            [720, "Spain", None, 45, 6, 70_000.0, 2, 1, 1, 90_000.0],
        ],
        columns=CANONICAL_FEATURE_ORDER,
    )


def test_complete_pipeline_owns_every_transformation_and_classifier():
    pipeline = build_model_pipeline(LogisticRegression(random_state=42))

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers[0][2] == NUMERIC_COLUMNS
    assert preprocessor.transformers[1][2] == CATEGORICAL_COLUMNS
    assert list(preprocessor.transformers[0][1].named_steps) == ["imputer", "scaler"]
    assert list(preprocessor.transformers[1][1].named_steps) == ["imputer", "encoder"]
    assert preprocessor.transformers[1][1].named_steps["encoder"].handle_unknown == "ignore"


def test_fitted_pipeline_imputes_and_is_independent_of_input_column_order():
    frame = training_frame()
    pipeline = build_model_pipeline(LogisticRegression(random_state=42)).fit(
        frame, [0, 1, 0, 1]
    )
    record = frame.iloc[[0]].copy()
    record["Geography"] = "Valid New Country"

    expected = pipeline.predict_proba(record)[:, 1]
    shuffled = pipeline.predict_proba(record[CANONICAL_FEATURE_ORDER[::-1]])[:, 1]

    np.testing.assert_allclose(expected, shuffled)
    assert list(
        pipeline.named_steps["preprocessor"].get_feature_names_out()
    ) == [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain",
        "Gender_Female",
        "Gender_Male",
    ]


def test_prediction_adapter_uses_one_loaded_pipeline_for_class_and_probability(
    tmp_path, monkeypatch
):
    schema = build_model_schema()
    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")

    class FakeFittedPipeline:
        classes_ = np.array([0, 1])

        def __init__(self):
            self.frames = []

        def predict(self, frame):
            self.frames.append(frame)
            return np.array([1])

        def predict_proba(self, frame):
            self.frames.append(frame)
            return np.array([[0.2, 0.8]])

    fitted = FakeFittedPipeline()
    load_calls = []

    def fake_load_object(file_path):
        load_calls.append(file_path)
        return fitted

    monkeypatch.setattr(prediction_pipeline, "load_object", fake_load_object)
    adapter = prediction_pipeline.PredictPipeline(str(tmp_path))
    frame = training_frame().iloc[[0]]
    labels, probabilities = adapter.predict(frame)

    assert labels.tolist() == [1]
    assert probabilities.tolist() == [0.8]
    assert len(load_calls) == 1
    assert fitted.frames[0] is frame
    assert fitted.frames[1] is frame


def test_versioned_schema_documents_every_canonical_feature():
    schema = build_model_schema()

    assert schema["schema_version"] == "1.0.0"
    assert schema["canonical_feature_order"] == CANONICAL_FEATURE_ORDER
    assert [feature["name"] for feature in schema["features"]] == CANONICAL_FEATURE_ORDER
    for feature in schema["features"]:
        assert {"data_type", "required", "nullable", "feature_type"} <= set(feature)
        assert feature["required"] is True
        assert feature["nullable"] is False
        assert feature["transformation_owner"].startswith("pipeline.preprocessor.")
