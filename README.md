# glucose-data-analysis
<p> <img src="Project_Logo/diabeatit-low-resolution-color-logo.png" width="400" height="300"/></p>

## By: Karina Kanjaria, Katie O'laughlin, Leslie Joe, Carlos Monsivais
Continuous glucose data analysis for blood glucose levels and glycemic events

### How to Run

### Data Structure
* Data_Pipeline
    1. sklearn_pipeline.py:

        Reads in Data rae data from either a PySpark Dataframe or Pandas Dataframe to run it through a sklearn pipeline where numerical 
        features are standardized, imputations are made, and one-hot-encoding > is done on categorical features. Within the 
        class called Sklearn_Pipeline there is a function for a Pyspark Dataframe called pyspark_sklearn_pipeline() and a function
        for a Pandas Dataframe called pandas_transform_features()

* Data_Schema
    1. schema.py:
    
        Stores data schemas used for reading in data either as a PySpark Dataframe or a Pandas Dataframe(data type and datetime columns are taken care of).
        Within this file the schemas to use for the output in PySpark when using Pandas User Defined Functions is used.

* EDA
    1. Glucose.ipynb, KarinaEDA.ipynb, LeslieEDA.ipynb:

        EDA notebooks.

* preprocessing
    1. wrappers.py:

        Initial data wrappers.

* Project_Logo
    1. diabeatit-low-resolution-color-logo.png:

        Logo for our group DiabeatIt!

* Read_In_Data
    1. read_data.py:
        Reads in data in both the Pyhton and PySpark format byusing schema types from Data_Schema folder.

* Time_Series_Features
    1. time_series_feature_creation.py:
        These are the wrapper functions that are used for feature engineering which include Multifractal Data Analysis, Poincare Analysis, Functional Principal Component Analysis, and Entropy Analysis