import os
import sys
from dataclasses import dataclass

import pandas as pd                                                                  # type: ignore
import numpy as np                                                                   # type: ignore
import seaborn as sns                                                                # type: ignore

from sklearn.linear_model import LinearRegression, Ridge , Lasso                     # type: ignore
from sklearn.tree import DecisionTreeRegressor                                       # type: ignore
from sklearn.svm import SVR                                                          # type: ignore
from sklearn.neighbors import KNeighborsRegressor                                    # type: ignore
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor , GradientBoostingRegressor # type: ignore
from catboost import CatBoostRegressor                                               # type: ignore
from xgboost import XGBRegressor                                                     # type: ignore

from sklearn.metrics import r2_score                                                 # type: ignore

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object , evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self , train_array , test_array ):
        try:
            logging.info("Split training and test input data")
            X_train , y_train , X_test , y_test = (
                train_array[: , :-1],
                train_array[: , -1],
                test_array[: , :-1],
                test_array[: , -1]
            )

            models = {
                "Linear Model" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "SMV" : SVR(),
                "K-Neighbor Regressor" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "XG Boost" : XGBRegressor(),
                "Cat Boost" : CatBoostRegressor(verbose=False),
                "Ada Boost" : AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train , y_train=y_train , X_test=X_test , y_test=y_test , models=models)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and teesting dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_scored = r2_score(y_test , predicted)

            return r2_scored


        except Exception as e:
            raise CustomException(e , sys)
