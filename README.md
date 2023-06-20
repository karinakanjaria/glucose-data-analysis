# glucose-data-analysis
<p> <img src="diabetitlogo.png" width="174" height="200"/></p>

# Long Term Horizon Predictions and Feature Explainability of Time Series Continuous Glucose Monitor Data
## by Katie O'Laughlin†, Carlos Monsivais†, Leslie Joe†, Karina Kanjaria†, Jamie Burks, and Benjamin Smarr, PhD
† These authors contributed equally to this work and share first authorship.

<hr>
<br>


## About

Over 11% of the US population has been diagnosed with diabetes, with millions experiencing a myriad of other health complications as a result. Luckily, diabetes can be treated and managed with the proper knowledge and tools. Through the use of data science and machine learning techniques, the DiabeatIt project seeks to help diabetes patients by analyzing what elements of their daily habits and characteristics contribute to hyper- and hypoglycemic events. Statistical and structural features are computed using 800 million time-series glucose measurements provided by Dexcom, and applied to several machine learning models in order to understand links between glycemic events and biological rhythms. Ultimately, an XGBoost decision tree classification model is implemented for feature explainability. This model achieved an accuracy of 61.2% with hyperparameter tuning. With such a model and its accompanying front-end application, patients and healthcare professionals are able to see which features most impacted the model’s predictions. This grants users the abilities to assess, understand, and potentially take actionable steps to improve their health.

(This is a completed Capstone project from the University of California, San Diego's M.A.S. Data Sciencea and Engineering program, 2023.)

<br>
<hr>
<br>

## How to Run

Please note this repository was built to pull from an existing databank of Dexcom's licenced data. If you do not have the same permissions, this code will not run.

1. Run the *create_env.sh* file by using the following command in teminal: `./create_env.sh`
2. Use one of the two *main* files:
    * Activate the environment by using the command `source glucose_venv/bin/activate` and then run *main.py* by using the command `python main.py`
    * Open *main.ipynb* and select the *glucose_venv* kernel to run each cell

<br>
<hr>
<br>