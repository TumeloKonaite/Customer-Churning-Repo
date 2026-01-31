import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    encoder_obj_file_path: str = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
    schema_file_path: str = os.path.join(ARTIFACTS_DIR, "schema.json")
    feature_columns_file_path: str = os.path.join(ARTIFACTS_DIR, "feature_columns.json")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = "Exited"
        self.numerical_columns = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]
        self.categorical_columns = ["Geography", "Gender"]

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Drop identifier columns if present
            drop_cols = ["RowNumber", "CustomerId", "Surname"]
            train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
            test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

            target_column_name = self.target_column_name
            feature_columns = self.numerical_columns + self.categorical_columns

            missing_cols = [col for col in feature_columns if col not in train_df.columns]
            if missing_cols:
                raise CustomException(
                    f"Missing required columns for transformation: {missing_cols}", sys
                )

            input_feature_train_df = train_df[feature_columns].copy()
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df[feature_columns].copy()
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying StandardScaler and OneHotEncoder as per notebook.")

            imputer = SimpleImputer(strategy="median")
            input_feature_train_df[self.numerical_columns] = imputer.fit_transform(
                input_feature_train_df[self.numerical_columns]
            )
            input_feature_test_df[self.numerical_columns] = imputer.transform(
                input_feature_test_df[self.numerical_columns]
            )

            scaler = StandardScaler()
            train_num = scaler.fit_transform(
                input_feature_train_df[self.numerical_columns]
            )
            test_num = scaler.transform(input_feature_test_df[self.numerical_columns])

            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_cat = encoder.fit_transform(
                input_feature_train_df[self.categorical_columns]
            )
            test_cat = encoder.transform(input_feature_test_df[self.categorical_columns])

            input_feature_train_arr = np.hstack([train_num, train_cat])
            input_feature_test_arr = np.hstack([test_num, test_cat])

            feature_names = list(self.numerical_columns) + list(
                encoder.get_feature_names_out(self.categorical_columns)
            )

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving scaler, encoder, and schema artifacts.")

            os.makedirs(os.path.dirname(self.data_transformation_config.schema_file_path), exist_ok=True)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler,
            )

            save_object(
                file_path=self.data_transformation_config.encoder_obj_file_path,
                obj=encoder,
            )

            schema = {
                "num_cols": self.numerical_columns,
                "all_cols": feature_columns,
            }
            with open(self.data_transformation_config.schema_file_path, "w") as f:
                json.dump(schema, f)

            with open(
                self.data_transformation_config.feature_columns_file_path, "w"
            ) as f:
                json.dump(feature_names, f)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
