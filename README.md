
# Census Income Prediction - Flask Application

This repository contains a Flask-based web application that predicts whether a person’s income exceeds 50K USD based on census data attributes. The model has been built by Egglisten Samuel, a Data Scientist, and uses a custom pipeline for data processing and prediction.

The web app allows users to input personal and professional data and receive a prediction of whether their income is greater than 50K USD or less than or equal to 50K USD.


## Table of Contents

* Project Overview
* Features
* Tech Stack
* Installation
* Usage
* Input Attributes
* Model Explanation
* Contributing
* License
## Project Overview
The Census Income Prediction app allows users to submit various demographic and professional information, such as age, education, marital status, occupation, hours worked per week, etc. Based on this information, the application predicts whether the person’s income is above 50K USD or below using a machine learning model.
## Features
#### Attributes Used in Prediction
The following attributes are used to make the prediction:

* Age: The age of the person.
* Capital Gain: Income from investments.
* Capital Loss: Losses from investments.
* Workclass: The type of employment.
* Education: The highest level of education achieved.
* Marital Status: Marital status (e.g., single, married).
* Occupation: Job type (e.g., tech, sales).
* Hours per week: The number of hours worked per week.
* Sex: Gender.
* Relationship: Relationship status.
* Race: Racial background.

#### Model
The model used in this project is a custom machine learning pipeline, built in Python, for predicting whether the income exceeds 50K USD.

The PredictionPipeline class is used to:

* Transform the input data into a format suitable for the model.
* Perform predictions based on the trained model.

The model has been trained using a census income dataset, where the task is to predict whether a person earns more than 50K USD per year based on demographic features.
## Tech Stack
* Frontend:
  * HTML 
  * CSS
* Backend:
    * Python
    * Flask (for serving the web application)
    * Scikit-learn (for machine learning model implementation)
    * NumPy (for numerical operations)
    * Pandas (for data handling)
* Others:
  * Git (for version control)
  * Virtualenv (for Python environment isolation)
## Installation

## 1. Clone the repository

```bash
git clone https://github.com/esamuel-91/CENSUS-INCOME_PREDICTION.git
cd CENSUS-INCOME_PREDICTION


```

## 2. Set up a Virtual Environment

```bash
conda create -p ./venv python=3.9 -c conda-forge
conda activate venv/

```

## 3. Install the required dependencies
```bash
pip install -r requirements.txt

```
## 4. Run the Flask Application

```bash
python app.py

```
## Usage
Once the dependencies are installed, you can run the Flask web application locally.

```bash
python app.py

```
The application will run on http://127.0.0.1:5001/ by default.


1. Navigate to the homepage:
Open your browser and visit http://127.0.0.1:5001/. You will be directed to the home page of the web application.

2. Submit data:
On the homepage, click on the Start Prediction button to access the form where you can enter various demographic and professional details.

3. Receive a prediction:
Once the form is submitted, the application will display the income prediction: either >50K or <=50K.

## Input Attributes (Example)
Input form (form.html):
* Age: 45
* Capital Gain: 2500
* Capital Loss: 0
* Workclass: Private
* Education: Bachelors
* Marital Status: Married
* Occupation: Exec-managerial
* ours per week: 40
* Sex: Male
* Relationship: Husband
* Race: White

## Output (result.html):
* Income Prediction: >50K
## Authors

- [@EgglistenSamuel](https://www.github.com/esamuel-91)