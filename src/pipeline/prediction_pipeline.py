import json
import os
import sys

import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.scaler_path = os.path.join("artifacts", "preprocessor.pkl")
        self.encoder_path = os.path.join("artifacts", "encoder.pkl")
        self.schema_path = os.path.join("artifacts", "schema.json")
        self.feature_columns_path = os.path.join("artifacts", "feature_columns.json")

        with open(self.schema_path, "r") as f:
            schema = json.load(f)
        self.num_cols = schema.get("num_cols", [])
        self.all_cols = schema.get("all_cols", [])

        if os.path.exists(self.feature_columns_path):
            with open(self.feature_columns_path, "r") as f:
                self.final_features = json.load(f)
        else:
            self.final_features = None

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(file_path=self.model_path)
            scaler = load_object(file_path=self.scaler_path)
            encoder = load_object(file_path=self.encoder_path)

            df = features.copy()
            df[self.num_cols] = scaler.transform(df[self.num_cols])

            categorical_cols = [col for col in self.all_cols if col not in self.num_cols]
            df_cat = encoder.transform(df[categorical_cols])
            df_cat = pd.DataFrame(
                df_cat,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=df.index,
            )

            df_final = pd.concat([df[self.num_cols], df_cat], axis=1)

            if self.final_features is not None:
                df_final = df_final.reindex(columns=self.final_features, fill_value=0)

            preds = model.predict(df_final.values)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        credit_score: float,
        geography: str,
        gender: str,
        age: float,
        tenure: float,
        balance: float,
        num_of_products: float,
        has_cr_card: float,
        is_active_member: float,
        estimated_salary: float,
    ):
        self.credit_score = credit_score
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_of_products = num_of_products
        self.has_cr_card = has_cr_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.credit_score],
                "Geography": [self.geography],
                "Gender": [self.gender],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.num_of_products],
                "HasCrCard": [self.has_cr_card],
                "IsActiveMember": [self.is_active_member],
                "EstimatedSalary": [self.estimated_salary],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
