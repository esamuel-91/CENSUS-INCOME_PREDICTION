import os,sys
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.utils import save_object,eval_models
from dataclasses import dataclass
import mlflow
import traceback

@dataclass
class ModelTrainerConfig:
    model_file_path:str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,X_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {
                "LogisticRegression":LogisticRegression(C=0.4, max_iter= 200, penalty= 'l2',random_state= 10,solver= 'sag'),
                "DecisionTreeClassifier":DecisionTreeClassifier(criterion='entropy',max_depth=20,max_features='sqrt',min_samples_leaf=4,min_samples_split=2,splitter='best'),
                "SVC":SVC(C=2,gamma='scale',kernel= 'rbf'),
                "RandomForestClassifier":RandomForestClassifier(criterion='entropy', max_depth=50, min_samples_leaf=3,min_samples_split=4, n_estimators=200)
            }

            report= eval_models(X_train,X_test,y_train,y_test,models=models)
            best_model_name = max(report,key=lambda model_name:report[model_name]["Accuracy"])
            best_model = models[best_model_name]
            best_model_score = report[best_model_name]["Accuracy"]

            print(f"Best Model found: {best_model} and best model score is: {best_model_score}")
            print("\n---------------------------------------------------------------------------------------------------\n")

            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=best_model
            )

            with mlflow.start_run():
                mlflow.log_artifact(self.trainer_config.model_file_path,"Model Pickle File")
                mlflow.log_param("Best Model",best_model)
                mlflow.log_param("Best Model Score",best_model_score)

        except Exception as e:
            with mlflow.start_run():
                mlflow.log_param("Exception Message",str(e))
                mlflow.log_param("Traceback",traceback.format_exc())
            raise CustomException(e,sys)