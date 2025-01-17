import os,sys
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
import mlflow

@dataclass
class DataingestionConfig:
    train_path:str = os.path.join("artifacts","train.csv")
    test_path:str = os.path.join("artifacts","test.csv")
    raw_path:str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataingestionConfig()

    def initiate_ingestion(self):
        try:
            #reading the dataset
            df = pd.read_csv(os.path.join("notebook","data/census_income.csv"))
            os.makedirs(os.path.dirname(self.ingestion_config.raw_path),exist_ok=True)

            #saving the dataframe in raw artifact
            df.to_csv(self.ingestion_config.raw_path,index=False)

            col_to_operate = ["workclass","occupation","income"]
            char_to_remove = ["?"," ?"]

            for char in char_to_remove:
                for col in col_to_operate:
                    if col == "income":
                        df[col] = df[col].str.replace(".","")
                    else:
                        df[col] = df[col].replace(char,np.nan)
            #dropping duplicates 
            df.drop_duplicates(keep="first",inplace=True)

            ## TRAIN TEST SPLIT 
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=42)
            train_set.to_csv(self.ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_path,index=False,header=True)

            with mlflow.start_run():
                mlflow.log_artifact(self.ingestion_config.train_path,"Train Artifact")
                mlflow.log_artifact(self.ingestion_config.test_path,"Test Artifact")
                mlflow.log_artifact(self.ingestion_config.raw_path,"Raw Artifact")
            
            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )
        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Exception Message",str(e))
            raise CustomException(e,sys)