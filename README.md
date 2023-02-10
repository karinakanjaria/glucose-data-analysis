# glucose-data-analysis
## By: Karina Kanjaria, Katie O'laughlin, Leslie Joe, Carlos Monsivais
Continuous glucose data analysis for blood glucose levels and glycemic events

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
