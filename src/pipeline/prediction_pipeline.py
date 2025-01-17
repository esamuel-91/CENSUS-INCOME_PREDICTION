from src.utils import load_obj
import os,sys
from src.exception import CustomException
import mlflow
import traceback
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_file_path = os.path.join("artifacts","preprocessor.pkl")
            model_file_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_obj(preprocessor_file_path)
            model = load_obj(model_file_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Exception Message",str(e))
                mlflow.log_param("Traceback",traceback.format_exc())
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                age:float,
                capital_gain:float,
                capital_loss:float,
                hours_per_week:float,
                workclass:object,
                education:object,
                marital_status:object,
                occupation:object,
                relationship:object,
                race:object,
                sex:object):
        self.age = age
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex


    def gather_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "age":[self.age],
                "capital-gain":[self.capital_gain],
                "capital-loss":[self.capital_loss],
                "hours-per-week":[self.hours_per_week],
                "workclass":[self.workclass],
                "education":[self.education],
                "marital-status":[self.marital_status],
                "occupation":[self.occupation],
                "relationship":[self.relationship],
                "race":[self.race],
                "sex":[self.sex]
            }

            #coverting dataset as dataframe
            df = pd.DataFrame(custom_data_dict)

            return df 
        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Exception Message",str(e))
                mlflow.log_param("Traceback",traceback.format_exc())
            raise CustomException(e,sys)
        