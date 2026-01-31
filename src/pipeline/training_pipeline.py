import sys

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run(self):
        try:
            logging.info("Starting training pipeline")
            train_data, test_data = DataIngestion().initiate_data_ingestion()
            train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
                train_data, test_data
            )
            score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
            logging.info("Training pipeline completed")
            return score
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    print(pipeline.run())
