"""Schema extraction from a fitted churn pipeline."""

from typing import Any

from src.components.data_transformation import ChurnModelPipeline
from src.model_schema import CATEGORICAL_COLUMNS, build_model_schema


def build_fitted_model_schema(pipeline: ChurnModelPipeline) -> dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categories = {
        column: [item.item() if hasattr(item, "item") else item for item in values]
        for column, values in zip(CATEGORICAL_COLUMNS, encoder.categories_)
    }
    return build_model_schema(
        known_categories=categories,
        transformed_feature_names=list(preprocessor.get_feature_names_out()),
    )
