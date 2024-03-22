import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw_data.csv")
    train_data_path = os.path.join("artifacts", "train_data.csv")
    test_data_path = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv("notebooks\data\gemstone.csv")
            logging.info("Read the data from csv file.")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Created the new data file.")

            logging.info("splitting the data into train and test set.")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Data splitting is done.")

            train_data.to_csv(self.ingestion_config.train_data_path, index = False)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False)
            logging.info("Created the train and test data files.")
            logging.info("Data ingestion completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured while ingesting data.")
            raise CustomException(e, sys)
            



