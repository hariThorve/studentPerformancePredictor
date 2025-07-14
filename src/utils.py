import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def saveObj(filePath, obj):
    try:
        dirPath = os.path.dirname(filePath)
        os.makedirs(dirPath, exist_ok=True)
        with open(filePath, "wb") as fileObj:
            dill.dump(obj, fileObj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluateModels(X_train,X_test,y_train,y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)


            yTrainpred = model.predict(X_train)
            yTestpred = model.predict(X_test)

            trainModelScore = r2_score(y_train, yTrainpred)
            testModelScore = r2_score(y_test, yTestpred)

            report[list(models.keys())[i]] = testModelScore

            return report


    except Exception as e:
        raise CustomException(e, sys)

def loadObj(filePath):
    try:
        with open(filePath, "rb") as fileObj:
            return dill.load(fileObj)
    except Exception as e:
        raise CustomException(e, sys)
