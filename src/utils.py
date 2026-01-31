import os
import sys
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for name, model in models.items():
            params = param.get(name, {})

            if params:
                gs = GridSearchCV(
                    model,
                    params,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            if hasattr(best_model, "predict_proba"):
                y_test_scores = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_test_scores = best_model.decision_function(X_test)
            else:
                y_test_scores = best_model.predict(X_test)

            test_model_score = roc_auc_score(y_test, y_test_scores)

            report[name] = test_model_score
            trained_models[name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
