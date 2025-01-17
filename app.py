from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
import mlflow,sys
import traceback
from src.exception import CustomException

application = Flask(__name__)
app = application

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/predict",methods = ["GET","POST"])
def predict():
    try:
        if request.method == "GET":
            return render_template("form.html")
        else:
            data = CustomData(
                age=int(request.form.get("age")),
                capital_gain=int(request.form.get("capital-gain")),
                capital_loss=int(request.form.get("capital-loss")),
                workclass=request.form.get("workclass"),
                education=request.form.get("education"),
                marital_status=request.form.get("marital-status"),
                occupation=request.form.get("occupation"),
                hours_per_week=int(request.form.get("hours-per-week")),
                sex=request.form.get("sex"),
                relationship=request.form.get("relationship"),
                race=request.form.get("race")
            )

            final_new_data = data.gather_data_as_dataframe()
            prediction_pipeline = PredictionPipeline()
            pred = prediction_pipeline.predict(final_new_data)

            results = round(pred[0])

            income_prediction = ">50K" if results == 1 else "<=50K"

            return render_template("result.html",final_result=income_prediction)
    except Exception as e:
        with mlflow.start_run():
            mlflow.log_param("Exception Message",str(e))
            mlflow.log_param("Traceback",traceback.format_exc())
        raise CustomException(e,sys)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001,debug=True)