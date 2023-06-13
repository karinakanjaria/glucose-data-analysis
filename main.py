from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      inter_train_location, inter_test_location, inter_val_location,\
                                      one_hot_encoding_data, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper, \
                                      model_storage_location, random_seed, \
                                      time_series_lag_values_created, \
                                      evaluation_metrics_output_storage, \
                                      feature_importance_storage_location, \
                                      overall_feature_importance_plot_location

from Data_Schema.schema import Pandas_UDF_Data_Schema
from Data_Generation.save_train_test_val import Create_Parquet_Files
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.imputation_pipeline import Date_And_Value_Imputation


from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.lag_features import Create_Lagged_Features
from Feature_Generation.time_series_feature_creation import TS_Features
from Feature_Generation.difference_features import Difference_Features

from Data_Pipeline.encoding_scaling_pipeline import Feature_Transformations

from Model_Creation.pyspark_xgboost import Create_PySpark_XGBoost

from Model_Predictions.pyspark_model_preds import Model_Predictions

from Model_Evaluation.pyspark_model_eval import Evaluate_Model

from Feature_Importance.model_feature_importance import Feature_Importance

from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot

import os


# PySpark UDF Schema Activation
pandas_udf_data_schema=Pandas_UDF_Data_Schema()

# Data Location
reading_data=Reading_Data()

# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()

# Create and clean parquet files from CSVs
create_parquet_files = Create_Parquet_Files()

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

# Classification Evaluation
evaluate_model=Evaluate_Model()

# Model Plots Feature Importance
xgboost_classification_plot=XGBoost_Classification_Plot()

# Feature Transformations
feature_transformations=Feature_Transformations()


pyspark_custom_imputation_schema=pandas_udf_data_schema.custom_imputation_pyspark_schema()


model_predictions=Model_Predictions()

# Feature Importance
feature_importance=Feature_Importance()

####### PySpark

create_parquet_files.train_val_test_step1(csv_files_location="/cephfs/data",
                                          checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet")
create_parquet_files.train_val_test_step2(checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet",
                                          cohort_location="/cephfs/data/cohort.csv")
create_parquet_files.train_val_test_step3(checkpoint_location="/cephfs/train_test_val/_checkpoint.parquet",
                                         train_location="/cephfs/train_test_val/train_set/",
                                         val_location="/cephfs/train_test_val/val_set/",
                                         test_location="/cephfs/train_test_val/test_set/")
create_parquet_files.train_val_test_step4(train_location="/cephfs/train_test_val/train_set/")


pyspark_df=reading_data.read_in_pyspark()


from pyspark.sql.functions import date_trunc, col

# 
pyspark_df=pyspark_df.withColumn("GlucoseDisplayTime", date_trunc("minute", col("GlucoseDisplayTime")))


pyspark_df=pyspark_df.distinct()


pyspark_df=pyspark_df.orderBy("PatientId", 
                              "GlucoseDisplayTime",
                              ascending=True)

###### Features Creation #######

data_types = ['train', 'test', 'val']

## data creation for train, test, and val datasets

for dataloc in data_types:

    # check if interpolation is complete
    interpolation_complete = os.path.exists('/cephfs/interpolation/' + dataloc)
    if interpolation_complete == False:
        # if not, create interpolated data and save
        date_and_value_imputation.interpolation_creation(dataloc)


    # read in interpolation data
    custom_imputation_pipeline = date_and_value_imputation.read_interpolation('/cephfs/interpolation/' + dataloc)


    # create difference features (firstDif, secondDif)
    df_differences = difference_features.add_difference_features(custom_imputation_pipeline)
    
    # add chunk values so grouping by day can be used in complex features and summary stats
    df_chunks = summary_stats_features.create_chunk_col(df_differences, chunk_val = 288)

    #check if poincare has been performed
    poincare_complete = os.path.exists('/cephfs/featuresData/poincare/' + dataloc)
    if poincare_complete == False:
        # create poincare values and save
        df_poincare = df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.poincare)
        df_poincare.repartition('NumId').write.parquet('/cephfs/featuresData/poincare/' + dataloc)
    else:
        # if created, read in
        df_poincare = spark.read.parquet('/cephfs/featuresData/poincare/' + dataloc)

    # check if entropy is created
    entropy_complete = os.path.exists('/cephfs/featuresData/entropy/' + dataloc)
    if entropy_complete == False:
        # if not, create entropy data and save
        df_entropy = df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.entropy)
        df_entropy.repartition('NumId').write.parquet('/cephfs/featuresData/entropy/' + dataloc)
    else:
        #read in entropy data
        df_entropy = spark.read.parquet('/cephfs/featuresData/entropy/' + dataloc)

    # merge poincare and entropy together to make complex feature df
    df_complex_features = df_poincare.join(df_entropy,['NumId', 'Chunk'])

    # if summary stats have been performed
    summary_stats_complete = os.path.exists('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
    if summary_stats_complete == False:
        # create summary statistics
        features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=df_chunks)
        features_summary_stats.repartition('NumId').write.parquet('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
        
    # if summary stats exist?
    final_df_exists = os.path.exists('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
    if final_df_exists == False:
        # if not read them in
        features_summary_stats=reading_data.read_in_pyspark_data_for_summary_stats('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')
        # join with complex features
        df_final = df_complex_features.join(features_summary_stats,['NumId', 'Chunk'])
        # save
        df_final.repartition('NumId').write.parquet('/cephfs/summary_stats/all_' + dataloc + '_bool_updated')


training_summary = spark.read.parquet('/cephfs/summary_stats/all_train_bool')
test_summary = spark.read.parquet('/cephfs/summary_stats/all_test_bool')
val_summary = spark.read.parquet('/cephfs/summary_stats/all_val_bool')