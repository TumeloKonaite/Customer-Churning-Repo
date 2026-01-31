import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


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

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
