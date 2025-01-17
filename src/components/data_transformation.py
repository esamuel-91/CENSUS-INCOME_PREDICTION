import os,sys
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from dataclasses import dataclass
import pandas as pd
import numpy as np
import mlflow
import traceback

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformation_objects(self):
        try:
            #segregating numerical and categorical columns 
            numerical_columns = ['age', 'capital-gain', 'capital-loss','hours-per-week']
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex']

            #setting up the pipeline 
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder(sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Exception Message",str(e))
                mlflow.log_param("Traceback",traceback.format_exc())
            raise CustomException(e,sys)
        
    def initiate_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #handling values for target columns 
            le = LabelEncoder()
            train_df["income"] = le.fit_transform(train_df["income"])
            test_df["income"] = le.transform(test_df["income"])

            target = "income"
            drop_column = [target,"fnlwgt","education-num","native-country"]

            input_feature_train_df = train_df.drop(drop_column,axis=1)
            input_target_train_df = train_df[target]

            input_feature_test_df = test_df.drop(drop_column,axis=1)
            input_target_test_df = test_df[target]

            ## obtaining preprocessor object
            preprocessor_obj = self.get_transformation_objects()

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(input_target_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_target_test_df)]

            save_object(
                file_path=self.transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )

            with mlflow.start_run():
                mlflow.log_artifact(self.transformation_config.preprocessor_file_path,"Preprocessor Pickle File")

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Excception Message",str(e))
                mlflow.log_param("Traceback",traceback.format_exc())
            raise CustomException(e,sys)