import sys
import pandas as pd
from src.exception import CustomException
from src.utils import loadObj


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            modelPath = "artifacts/model.pkl"
            preprocessorPath = "artifacts/perprocessor.pkl"
            model = loadObj(filePath = modelPath)
            preprocessor = loadObj(filePath = preprocessorPath)

            # scailing the data coming from the frontend using the before created pickle file

            scaledData = preprocessor.transform(features)

            #predicting the data

            prediction = model.predict(scaledData)

            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender: str,
        raceEthnicity: str,
        parentalLevelofEducation,
        lunch:str,
        testPreparationCourse:str,
        readingScore:int,
        writingScore:int
    ):
        self.gender = gender
        self.raceEthnicity = raceEthnicity
        self.parentalLevelofEducation = parentalLevelofEducation
        self.lunch = lunch
        self.testPreparationCourse = testPreparationCourse
        self.readingScore = readingScore
        self.writingScore = writingScore

    def getDataAsDataFrame(self):
        try:
            customDataInputDict = {
                "gender": [self.gender],
                "race_ethnicity": [self.raceEthnicity],
                "parental_level_of_education": [self.parentalLevelofEducation],
                "lunch":[self.lunch],
                "test_preparation_course": [self.testPreparationCourse],
                "reading_score": [self.readingScore],
                "writing_score":[self.writingScore]
            }
            return pd.DataFrame(customDataInputDict)
        except Exception as e:
            raise CustomException(e,sys)
