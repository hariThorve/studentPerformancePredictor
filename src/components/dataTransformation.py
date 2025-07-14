import os
from dataclasses import dataclass
import sys
import numpy as np
import pandas as pd
from src.utils import saveObj
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessorObjFilePath = os.path.join("artifacts", "perprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.dataTransoformationConfig = DataTransformationConfig()

    #function to create pickle file for data preprocessing where all categorical data will be converted to numerical data and other preprocessing or perform standard Scaler or OHE
    def getDataTransformerObj(self):
        try:
            numericalColumns = ["writing_score", "reading_score"]
            categoricalColumns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            numericalPipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )
            logging.info("Numerical values scaling done")

            categoricalPipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oneHotEncoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical values encoded and scaled")

            logging.info(f"Categorical Values {categoricalColumns}")
            logging.info(f"Numerical Values {numericalColumns}")


            preprocessor = ColumnTransformer(
                [
                ("numericalPipeline", numericalPipeline, numericalColumns),
                ("categoricalPipeline", categoricalPipeline, categoricalColumns)
                ]

            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initateDataTransformation(self, trainDataPath, testDataPath):
        try:

            trainDf = pd.read_csv(trainDataPath)
            testDf = pd.read_csv(testDataPath)

            logging.info("Read the Train and Test Data")

            logging.info("Obtaining preprocessor info/object")

            preprocessingObj = self.getDataTransformerObj()
            targetColumn = "math_score"
            # numericalColumns = ["writing_score", "reading_score"]

            inputFeatureTrain = trainDf.drop(columns=[targetColumn], axis=1)
            targetFeatureTrain = trainDf[targetColumn]

            inputFeatureTest = testDf.drop(columns=[targetColumn], axis=1)
            targetFeatureTest = testDf[targetColumn]

            logging.info("Applying preprocessing object on training dataFrame and testing dataframe")

            inputFeatureTrainArr = preprocessingObj.fit_transform(inputFeatureTrain)
            inputFeatureTestArr = preprocessingObj.transform(inputFeatureTest)

            trainArr = np.c_[inputFeatureTrainArr, np.array(targetFeatureTrain)]
            testArr = np.c_[inputFeatureTestArr, np.array(targetFeatureTest)]

            logging.info("Saved Processing Info")

            saveObj(
                filePath = self.dataTransoformationConfig.preprocessorObjFilePath,
                obj = preprocessingObj
            )
            return(
                trainArr,
                testArr,
                self.dataTransoformationConfig.preprocessorObjFilePath
            )



        except Exception as e:
            raise CustomException(e,sys)
