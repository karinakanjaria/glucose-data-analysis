from Input_Variables.read_vars import raw_data_storage, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper, \
                                      ml_models_train_split, ml_models_test_split, model_storage_location, \
                                      time_series_lag_values_created

from Data_Schema.schema import Pandas_UDF_Data_Schema
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.sklearn_pipeline import Sklearn_Pipeline
from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.lag_features import Create_Lagged_Features
from Model_Creation.xgboost_model import XGBoost_Classification
from Model_Evaluation.classification_evaluation import Classification_Evalaution_Metrics
from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot


# PySpark UDF Schema Activation
pandas_udf_data_schema=Pandas_UDF_Data_Schema()

# Data Location
reading_data=Reading_Data(data_location=raw_data_storage)

# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()

# Sklearn Pipeline
pandas_sklearn_pipeline=Sklearn_Pipeline()

# Features Daily Stats Module
summary_stats_features=Summary_Stats_Features()

# Features Lagged Value
create_lag_features=Create_Lagged_Features()

# XGBoost Model Module
xgboost_classification=XGBoost_Classification()

# Classification Evaluation
classification_evalaution_metrics=Classification_Evalaution_Metrics()

# Model Plots Feature Importance
xgboost_classification_plot=XGBoost_Classification_Plot()



####### PySpark
pyspark_df=reading_data.read_in_pyspark()


from pyspark.sql.functions import date_trunc, col
pyspark_df=pyspark_df.withColumn("GlucoseDisplayTime", date_trunc("minute", col("GlucoseDisplayTime")))


pyspark_df=pyspark_df.distinct()


pyspark_df=pyspark_df.orderBy("PatientId", 
                              "GlucoseDisplayTime",
                              ascending=True)

pyspark_df.show(5)



####### PySpark
pyspark_custom_imputation_schema=pandas_udf_data_schema.custom_imputation_pyspark_schema()
pyspark_custom_imputation_pipeline=pandas_sklearn_pipeline.\
                                    pyspark_custom_imputation_pipeline(df=pyspark_df,\
                                    output_schema=pyspark_custom_imputation_schema,\
                                    analysis_group=analysis_group)


pyspark_custom_imputation_pipeline.show(5)