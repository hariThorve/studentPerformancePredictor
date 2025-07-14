import os
import sys
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from src.utils import saveObj, evaluateModels

@dataclass
class ModelTrainerConfig:
    trainedModelFilePath = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()

    def initiateModelTrainer(self, trainingArr, testArr):
        try:
            logging.info("Splitting training and testing data")

            #assigning values according to the object returned from the dataTransformation obj
            X_train,y_train, X_test, y_test = (
                trainingArr[:,:-1],
                trainingArr[:,-1],
                testArr[:,:-1],
                testArr[:,-1]
            )

            #creating models dictionary
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRFRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            #hyper parameter tuning
            params={
                "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            modelReport:dict = evaluateModels(
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                models = models,
                param = params
            )

            bestModelScore = max(sorted(modelReport.values()))
            bestModelName = list(modelReport.keys())[
                list(modelReport.values()).index(bestModelScore)
            ]

            bestModel = models[bestModelName]

            if bestModelScore < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best model Found on training and test dataset is {bestModel}")

            saveObj(
                filePath=self.modelTrainerConfig.trainedModelFilePath,
                obj=bestModel
            )

            predicted=bestModel.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e, sys)
