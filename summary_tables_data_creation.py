from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper


from Data_Pipeline.imputation_pipeline import Date_And_Value_Imputation
from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.difference_features import Difference_Features

import os

# Data Location
date_and_value_imputation=Date_And_Value_Imputation()
# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()
# Features Daily Stats Module
summary_stats_features=Summary_Stats_Features()
# Features Differences
difference_features=Difference_Features()


training_files_directory=os.listdir('/cephfs/interpolation/train/')

training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]

# tester for now
training_files=training_files[0:1]

for training_file in training_files:
    # Read in Data
    training_custom_imputation_pipeline = date_and_value_imputation.read_interpolation(f'/cephfs/interpolation/train/{training_file}')
        
    training_df_added_binary_labels=create_binary_labels.pyspark_binary_labels(df=training_custom_imputation_pipeline)

    training_df_differences = difference_features.add_difference_features(training_df_added_binary_labels)
    
    training_df_chunks = summary_stats_features.create_chunk_col(training_df_differences, chunk_val = 288)
    
    training_features_summary_stats=summary_stats_features.pyspark_summary_statistics(df=training_df_chunks)
    training_features_summary_stats.show(5)