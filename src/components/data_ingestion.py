import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join("dataset", "Churn_Modelling.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            source_path = self.ingestion_config.source_data_path
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source dataset not found at {source_path}")

            df = pd.read_csv(source_path)
            logging.info(f"Read dataset from {source_path} with shape {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw dataset saved to {self.ingestion_config.raw_data_path}")

            logging.info("Train test split initiated")
            stratify_col = df["Exited"] if "Exited" in df.columns else None
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=stratify_col
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
