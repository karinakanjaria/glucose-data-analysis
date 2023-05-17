from Input_Variables.read_vars import random_seed

from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.encoding_scaling_pipeline import Feature_Transformations
from Model_Creation.pyspark_xgboost import Create_PySpark_XGBoost

import os


reading_data=Reading_Data()
feature_transformations=Feature_Transformations()
create_pyspark_xgboost=Create_PySpark_XGBoost()

training_files_directory=os.listdir('/cephfs/summary_stats/train')
training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]

total_file_iteration=len(training_files)
iteration=1

# tester for now
training_files=training_files[0:3]

model_storage_location='/cephfs/Model_Summary_Stats'

for file in training_files:
    # Read in Summary Statistics
    summary_stats=reading_data\
    .read_in_pyspark_data_for_summary_stats(data_location=f'/cephfs/summary_stats/train/{file}')
    
    # Numerical Transformation Stages
    training_numerical_stages=feature_transformations.numerical_scaling(df=summary_stats)

    if iteration==1:
        # XGBoost Model
        xgboost_model=create_pyspark_xgboost\
        .initial_training_xgboost_regression(ml_df=training_features_summary_stats,
                                             stages=training_numerical_stages,
                                             model_storage_location=model_storage_location,
                                             random_seed=random_seed)
    else:
        
        
    print(f'Completed: {iteration}/{total_file_iteration}')
    iteration=iteration+1