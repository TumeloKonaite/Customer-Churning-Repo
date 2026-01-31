import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.metrics import compute_classification_metrics, lift_curve
from src.utils import evaluate_models, save_object


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(ARTIFACTS_DIR, "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0],
                },
                "Decision Tree": {
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise CustomException("Model evaluation did not return results.")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best model: {best_model_name} with ROC-AUC {best_model_score:.3f}"
            )

            if hasattr(best_model, "predict_proba"):
                y_test_scores = best_model.predict_proba(X_test)[:, 1]
                threshold = 0.5
            elif hasattr(best_model, "decision_function"):
                y_test_scores = best_model.decision_function(X_test)
                threshold = 0.0
            else:
                y_test_scores = best_model.predict(X_test)
                threshold = 0.5

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            metrics = compute_classification_metrics(
                y_true=y_test, y_score=y_test_scores, threshold=threshold
            )
            metrics["roc_auc"] = float(best_model_score)
            metrics["lift_curve"] = lift_curve(y_test, y_test_scores)

            return {
                "best_model_name": best_model_name,
                "best_model_score": float(best_model_score),
                "metrics": metrics,
                "model_path": self.model_trainer_config.trained_model_file_path,
            }

        except Exception as e:
            raise CustomException(e, sys)
