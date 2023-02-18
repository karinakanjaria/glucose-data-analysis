# glucose-data-analysis
<p> <img src="Project_Logo/diabeatit-low-resolution-color-logo.png" width="400" height="300"/></p>

## By: Karina Kanjaria, Katie O'laughlin, Leslie Joe, Carlos Monsivais
Continuous glucose data analysis for blood glucose levels and glycemic events

### How to Run
1. Run the create_env.sh file by using the foloowing command: ./create_env.sh
2. If not intiialized, activate teh environment by using the source glucose_venv/bin/activate
3. Run either the main.ipynb or main.py file to get output

### Data Structure
* Data_Pipeline
    1. sklearn_pipeline.py:

        Reads in raw data from either a PySpark Dataframe or Pandas Dataframe to run it through a sklearn pipeline where numerical 
        features are standardized, imputations are made, and one-hot-encoding > is done on categorical features. Within the 
        class called Sklearn_Pipeline there is a function for a Pyspark Dataframe called pyspark_sklearn_pipeline() and a function
        for a Pandas Dataframe called pandas_transform_features()

* Data_Schema
    1. schema.py:
    
        Stores data schemas used for reading in data either as a PySpark Dataframe or a Pandas Dataframe(data type and datetime columns are taken care of).
        Within this file the schemas to use for the output in PySpark when using Pandas User Defined Functions is used.
        
        1. Metadata - Patient information on Gender, DOB, Age, Diaebtes Type, and if they are receieving treatments
            - file: cohort.csv
            - columns: 'Unnamed', 'UserId', 'Gender', 'DOB', 'Age', 'DiabetesType', 'Treatment'
                      
       2. Glucose Data - Raw data of the glucose measurement at a specific time per patient  
       
           - file: glucose_record_{date}.csv
           - columns: 'PostDate', 'IngestionDate', 'PostId', 'PostTime', 'PatientId', 'Stream', 'SequenceNumber', 'TransmitterNumber', 'ReceiverNumber', 'RecordedSystemTime', 'RecordedDisplayTime', 'RecordedDisplayTimeRaw', 'TransmitterId', 'TransmitterTime', 'GlucoseSystemTime', 'GlucoseDisplayTime', 'GlucoseDisplayTimeRaw', 'Value', 'Status', 'TrendArrow', 'TrendRate', 'IsBackFilled', 'InternalStatus', 'SessionStartTime'
                      
       3. Glucose Data Dictionary - List of possible values per column of Glucose Data
           
           - file: glucose.json
           - columns: 'PostDate', 'IngestionDate', 'PostId', 'PostTime', 'PatientId', 'Stream', 'SequenceNumber', 'TransmitterNumber', 'ReceiverNumber', 'RecordedSystemTime',  'RecordedDisplayTime', 'RecordedDisplayTimeRaw', 'TransmitterId', 'TransmitterTime', 'GlucoseSystemTime', 'GlucoseDisplayTime', 'GlucoseDisplayTimeRaw', 'Value', 'Status', 'TrendArrow', 'TrendRate', 'IsBackFilled', 'InternalStatus', 'SessionStartTime'
            
        4. Glucose Data Column Definitions - Description of each column in raw Glucose Dataset 
        
            - file: glucose_records.json
            - columns: key, title, description

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

