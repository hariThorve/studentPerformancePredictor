from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.pipeline import METHODS
from sklearn.preprocessing import StandardScaler
from src.pipeline.predictionPipline import CustomData, PredictPipeline

app = Flask(__name__)

# home page routing

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predictDatapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            raceEthnicity=request.form.get('ethnicity'),
            parentalLevelofEducation=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            testPreparationCourse=request.form.get('test_preparation_course'),
            readingScore=float(request.form.get('reading_score')),
            writingScore=float(request.form.get('writing_score'))
            )
        pred_df=data.getDataAsDataFrame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="127.0.0.1", port=3000)
