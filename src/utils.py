import sys,os
from src.exception import CustomException
import mlflow
import pickle
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def save_object(file_path:str,obj):
    try:
        dir_path:str = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file=file_path,mode="wb") as file_obj:
            pickle.dump(file=file_obj,obj=obj)
    except Exception as e:
        with mlflow.start_run():
            mlflow.log_param("Exception Message",str(e))
        raise CustomException(e,sys)
    
def load_obj(file_path:str):
    try:
        with open(file=file_path,mode="rb") as file_obj:
            return pickle.load(file=file_obj)
    except Exception as e:
        with mlflow.start_run():
            mlflow.log_param("Exception Message",str(e))
        raise CustomException(e,sys)
    
def eval_models(X_train,X_test,y_train,y_test,models):
    try:
        report = {}

        for model_name,model in models.items():
            print(f"Evaluating the model: {model_name}")

            model.fit(X_train,y_train)

            #making prediction on the test data
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            class_report = classification_report(y_test,y_pred)
            confi_matrix = confusion_matrix(y_test,y_pred)

            report[model_name] = {
                "Accuracy":accuracy,
                "Classification Report":class_report,
                "Confusion Matrix":confi_matrix
            }

            return report
    except Exception as e:
        with mlflow.start_run():
            mlflow.log_param("Exception Message",str(e))
        raise CustomException(e,sys)