from src.components.dataIngestion import DataIngestion
from src.components.dataTransformation import DataTransformation
from src.components.modelTrainer import ModelTrainer

if __name__ == "__main__":
    # Data Ingestion
    obj = DataIngestion()
    trainData, testData = obj.initiateDataIngestion()

    # Data Transformation
    dataTransformationObj = DataTransformation()
    trainDataArr, testDataArr, _ = dataTransformationObj.initateDataTransformation(trainData, testData)

    # Model Training
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiateModelTrainer(trainDataArr, testDataArr))