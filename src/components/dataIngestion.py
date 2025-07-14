import os
import sys
from src.components import dataTransformation
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.dataTransformation import DataTransformation
from src.components.dataTransformation import DataTransformationConfig
from src.components.modelTrainer import ModelTrainerConfig
from src.components.modelTrainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    trainDatapath: str = os.path.join("artifacts", "train.csv")
    testDatapath: str = os.path.join("artifacts", "test.csv")
    rawDatapath: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            #making directories using class DataIngestion
            os.makedirs(os.path.dirname(self.ingestion_config.trainDatapath), exist_ok=True)

            #below line creates a copy of stud.csv in the folder mentioned in ingestion_config.rawDatapath
            df.to_csv(self.ingestion_config.rawDatapath,index=False, header=True)

            logging.info("Train Test Split Initiated")

            trainData, testData  = train_test_split(df, test_size=0.2, random_state=42)

            #below line creates a train data file in the location ingestion.trainDatapath

            trainData.to_csv(self.ingestion_config.trainDatapath, index=False, header=True)

            #below line creates a test data file in the location ingestion.testDatapath

            testData.to_csv(self.ingestion_config.testDatapath, index=False, header=True)

            logging.info("Ingestion of data is done.")

            return(
                self.ingestion_config.trainDatapath,
                self.ingestion_config.testDatapath
            )


        except Exception as e:
            raise CustomException(e,sys)

