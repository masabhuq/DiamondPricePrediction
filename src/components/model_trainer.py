import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModelTrainerConfig:
    # To add in the filepath of save_obj
    trained_model_file_path = os.path.join("artifacts", "model.joblib")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Separating features and target columns.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # all the columns up to last column
                train_array[:,-1], # last column
                test_array[:, :-1],
                test_array[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n=====================\n")
            logging.info(f"Model Report: {model_report}")

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[ #accessing names of the models
                list(model_report.values()).index(best_model_score) # index at which the best model score is found
            ]

            best_model = models[best_model_name]

            print(f"Best model found, Model Name: {best_model_name}, R2 Score = {best_model_score}")
            print("\n===================\n")
            logging.info(f"Best model found, Model Name: {best_model_name}, R2 Score = {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            logging.info("Exception occured during model training.")
            raise CustomException(e, sys)