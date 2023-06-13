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








###### Features Creation #######

interpolation_complete = os.path.exists('/cephfs/interpolation/train')
if interpolation_complete == False:
    date_and_value_imputation.interpolation_creation('train')


training_custom_imputation_pipeline = date_and_value_imputation.read_interpolation('/cephfs/interpolation/train/')
training_custom_imputation_pipeline.show(2)


interpolation_complete = os.path.exists('/cephfs/interpolation/test')
if interpolation_complete == False:
    date_and_value_imputation.interpolation_creation('test')

    
testing_custom_imputation_pipeline = date_and_value_imputation.read_interpolation('/cephfs/interpolation/test/')
testing_custom_imputation_pipeline.show(2)


interpolation_complete = os.path.exists('/cephfs/interpolation/val')
if interpolation_complete == False:
    date_and_value_imputation.interpolation_creation('val')

    
val_custom_imputation_pipeline = date_and_value_imputation.read_interpolation('/cephfs/interpolation/val/')
val_custom_imputation_pipeline.show(2)


training_df_differences = difference_features.add_difference_features(training_custom_imputation_pipeline)
training_df_differences.show(5)

training_df_chunks = summary_stats_features.create_chunk_col(training_df_differences, chunk_val = 288)
training_df_chunks.show(5)


testing_df_differences = difference_features.add_difference_features(testing_custom_imputation_pipeline)
testing_df_differences.show(5)

testing_df_chunks = summary_stats_features.create_chunk_col(testing_df_differences, chunk_val = 288)
testing_df_chunks.show(5)


val_df_differences = difference_features.add_difference_features(val_custom_imputation_pipeline)
val_df_differences.show(5)

val_df_chunks = summary_stats_features.create_chunk_col(val_df_differences, chunk_val = 288)
val_df_chunks.show(5)


poincare_complete = os.path.exists('/cephfs/featuresData/poincare/train')
if poincare_complete == False:
    training_df_poincare = training_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.poincare)
    training_df_poincare.repartition('NumId').write.parquet('/cephfs/featuresData/poincare/train')
else:
    training_df_poincare = spark.read.parquet('/cephfs/featuresData/poincare/train')
training_df_poincare.show(5)

entropy_complete = os.path.exists('/cephfs/featuresData/entropy/train')
if entropy_complete == False:
    training_df_entropy = training_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.entropy)
    training_df_entropy.repartition('NumId').write.parquet('/cephfs/featuresData/entropy/train')
else:
    training_df_entropy = spark.read.parquet('/cephfs/featuresData/entropy/train')
training_df_entropy.show(5)

training_df_complex_features = training_df_poincare.join(training_df_entropy,['NumId', 'Chunk'])
training_df_complex_features.show()


poincare_complete = os.path.exists('/cephfs/featuresData/poincare/test')
if poincare_complete == False:
    testing_df_poincare = testing_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.poincare)
    testing_df_poincare.repartition('NumId').write.parquet('/cephfs/featuresData/poincare/test')
else:
    testing_df_poincare = spark.read.parquet('/cephfs/featuresData/poincare/test')
testing_df_poincare.show(5)

entropy_complete = os.path.exists('/cephfs/featuresData/entropy/test')
if entropy_complete == False:
    testing_df_entropy = testing_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.entropy)
    testing_df_entropy.repartition('NumId').write.parquet('/cephfs/featuresData/entropy/test')
else:
    testing_df_entropy = spark.read.parquet('/cephfs/featuresData/entropy/test')
testing_df_entropy.show(5)

testing_df_complex_features = testing_df_poincare.join(testing_df_entropy,['NumId', 'Chunk'])
testing_df_complex_features.show()


poincare_complete = os.path.exists('/cephfs/featuresData/poincare/test')
if poincare_complete == False:
    testing_df_poincare = testing_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.poincare)
    testing_df_poincare.repartition('NumId').write.parquet('/cephfs/featuresData/poincare/test')
else:
    testing_df_poincare = spark.read.parquet('/cephfs/featuresData/poincare/test')
testing_df_poincare.show(5)

entropy_complete = os.path.exists('/cephfs/featuresData/entropy/val')
if entropy_complete == False:
    val_df_entropy = val_df_chunks.groupby(['NumId', 'Chunk']).apply(ts_features.entropy)
    val_df_entropy.repartition('NumId').write.parquet('/cephfs/featuresData/entropy/val')
else:
    val_df_entropy = spark.read.parquet('/cephfs/featuresData/entropy/val')
val_df_entropy.show(5)

val_df_complex_features = val_df_poincare.join(val_df_entropy,['NumId', 'Chunk'])
val_df_complex_features.show()


summary_stats_complete = os.path.exists('/cephfs/summary_stats/encoded/one_hot_train/summary_stats_cohort_bool_encoded.parquet')
if summary_stats_complete == False:
    training_features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=training_df_chunks)
else:
    training_features_summary_stats=reading_data.read_in_pyspark_data_for_summary_stats('/cephfs/summary_stats/encoded/one_hot_train/summary_stats_cohort_bool_encoded.parquet')

training_features_summary_stats.show(3)


summary_stats_complete = os.path.exists('/cephfs/summary_stats/encoded/one_hot_test/summary_stats_cohort_bool_encoded.parquet')
if summary_stats_complete == False:
    testing_features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=testing_df_chunks)
else:
    testing_features_summary_stats=reading_data.read_in_pyspark_data_for_summary_stats('/cephfs/summary_stats/encoded/one_hot_test/summary_stats_cohort_bool_encoded.parquet')

testing_features_summary_stats.show(3)


summary_stats_complete = os.path.exists('/cephfs/summary_stats/encoded/one_hot_val/summary_stats_cohort_bool_encoded.parquet')
if summary_stats_complete == False:
    val_features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=val_df_chunks)
else:
    val_features_summary_stats=reading_data.read_in_pyspark_data_for_summary_stats('/cephfs/summary_stats/encoded/one_hot_val/summary_stats_cohort_bool_encoded.parquet')

val_features_summary_stats.show(3)


final_train = os.path.exists('/cephfs/summary_stats/all_train_bool')
if final_train == False:
    training_df_final = training_df_complex_features.join(training_features_summary_stats,['NumId', 'Chunk'])
    training_df_final.repartition('NumId').write.parquet('/cephfs/summary_stats/all_train_bool')
else:
    training_df_final = spark.read.parquet('/cephfs/summary_stats/all_train_bool')
training_df_final.show(5)

final_test = os.path.exists('/cephfs/summary_stats/all_test_bool')
if final_test == False:
    testing_df_final = testing_df_complex_features.join(testing_features_summary_stats,['NumId', 'Chunk'])
    testing_df_final.repartition('NumId').write.parquet('/cephfs/summary_stats/all_test_bool')
else:
    testing_df_final = spark.read.parquet('/cephfs/summary_stats/all_test_bool')
testing_df_final.show(5)

final_val = os.path.exists('/cephfs/summary_stats/all_val_bool')
if final_val == False:
    val_df_final = val_df_complex_features.join(val_features_summary_stats,['NumId', 'Chunk'])
    val_df_final.repartition('NumId').write.parquet('/cephfs/summary_stats/all_val_bool')
else:
    val_df_final = spark.read.parquet('/cephfs/summary_stats/all_val_bool')
val_df_final.show(5)

