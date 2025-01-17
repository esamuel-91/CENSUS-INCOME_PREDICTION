from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_ingestion()
    transformation = DataTransformation()
    train_arr,test_arr=transformation.initiate_transformation(train_path=train_path,test_path=test_path)
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)