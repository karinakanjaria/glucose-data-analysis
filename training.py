from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      one_hot_encoding_data, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper, \
                                      model_storage_location, random_seed
from Data_Schema.schema import Pandas_UDF_Data_Schema
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.imputation_pipeline import Date_And_Value_Imputation
from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.lag_features import Create_Lagged_Features
from Feature_Generation.time_series_feature_creation import TS_Features
from Feature_Generation.difference_features import Difference_Features
from Data_Pipeline.encoding_scaling_pipeline import Feature_Transformations
from Model_Creation.pyspark_xgboost import Create_PySpark_XGBoost

import os


# PySpark UDF Schema Activation
pandas_udf_data_schema=Pandas_UDF_Data_Schema()
# Data Location
reading_data=Reading_Data()
# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()
# Imputation
date_and_value_imputation=Date_And_Value_Imputation()
# Features Daily Stats Module
summary_stats_features=Summary_Stats_Features()
# Features Complex
ts_features=TS_Features()
# Features Lagged Value
create_lag_features=Create_Lagged_Features()
# Features Differences
difference_features=Difference_Features()
# PySpark XGBoost Model Module
create_pyspark_xgboost=Create_PySpark_XGBoost()
# Feature Transformations
feature_transformations=Feature_Transformations()
pyspark_custom_imputation_schema=pandas_udf_data_schema.custom_imputation_pyspark_schema()

training_files_directory=os.listdir(train_data_storage)
training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]

# tester for now
training_files=training_files[0:3]

iteration=0

for training_file in training_files:
    # Read in Data
    training_df=reading_data.read_in_pyspark_data(data_location=train_data_storage)

    # Imputation
    training_custom_imputation_schema=pandas_udf_data_schema.custom_imputation_pyspark_schema()
    training_custom_imputation_pipeline=date_and_value_imputation.\
                                        pyspark_custom_imputation_pipeline(df=training_df,
                                                                        output_schema=pyspark_custom_imputation_schema,
                                                                        analysis_group=analysis_group)

    # Add Binary Labels
    training_df_added_binary_labels=create_binary_labels.pyspark_binary_labels(df=training_custom_imputation_pipeline)

    # Feature Creation
    training_df_differences = difference_features.add_difference_features(training_df_added_binary_labels)
    training_df_chunks = summary_stats_features.create_chunk_col(training_df_differences, chunk_val = 288)

    # Summary Stats
    training_features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=training_df_chunks)

    # Numerical Transformation Stages
    training_numerical_stages=feature_transformations.numerical_scaling(df=training_features_summary_stats)

    if iteration==0:
        # XGBoost Model
        xgboost_model=create_pyspark_xgboost.xgboost_classifier(ml_df=training_features_summary_stats,
                                                                stages=training_numerical_stages,
                                                                model_storage_location=model_storage_location,
                                                                random_seed=random_seed)
    else:
        
    
    iteration=iteration+1
    print(iteration)