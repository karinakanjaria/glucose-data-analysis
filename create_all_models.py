################################ Libraries ################################
from Input_Variables.read_vars import xgboost_regression_model_storage_location, \
                                      linear_regression_model_storage_location, \
                                      random_forest_regression_model_storage_location, \
                                      factorization_machines_regression_model_storage, \
                                      random_seed
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.scaling_pipeline import Feature_Transformations
from Model_Creation.regression_models import Create_Regression_Models
import os


################################ Read In Modules ################################
reading_data=Reading_Data()
feature_transformations=Feature_Transformations()
create_regression_models=Create_Regression_Models()


################################ Read In Data ################################
# Training Summary Stats Data
training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_train_bool_updated'),x),
                                               os.listdir('/cephfs/summary_stats/all_train_bool_updated')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]


# Cross Validation Summary Stats Data
val_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_val_bool_updated'), x),
                                          os.listdir('/cephfs/summary_stats/all_val_bool_updated')))
val_files=[i for i in val_files if not ('.crc' in i or 'SUCCESS' in i)]


# Calling DataFrames
summary_stats_train=reading_data.read_in_all_summary_stats(file_list=training_files)
summary_stats_val=reading_data.read_in_all_summary_stats(file_list=val_files)


################################ Combine Train and Cross Validation ################################
df_train_val_combined=summary_stats_train.union(summary_stats_val)
df_train_val_combined.show(2)
print((df_train_val_combined.count(), len(df_train_val_combined.columns)))


################################ Stages: Scaling Using Custom Transformer ################################
pipeline_transformation_stages=feature_transformations.numerical_scaling(df=df_train_val_combined)


################################ Regression Models ################################
regression_models_storage_locations=[xgboost_regression_model_storage_location, 
                                     linear_regression_model_storage_location,
                                     random_forest_regression_model_storage_location,
                                     factorization_machines_regression_model_storage]
create_regression_models\
        .regression_modeling(ml_df=df_train_val_combined,
                             stages=pipeline_transformation_stages, 
                             random_seed=random_seed,
                             regression_models_storage_locations=regression_models_storage_locations,
                             num_folds=3)


# ################################ Classification Models ################################
# create_pyspark_xgboost\
#         .regression_modeling(ml_df=summary_stats_train,
#                                              stages=pipeline_transformation_stages, 
#                                              random_seed=random_seed,
#                                              xgb_reg_model_storage_location=xgb_reg_model_storage_location,
#                                              k=k_folds)