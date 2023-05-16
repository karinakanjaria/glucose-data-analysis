from Input_Variables.read_vars import train_data_storage, validation_data_storage, test_data_storage, \
                                      analysis_group, \
                                      daily_stats_features_lower, daily_stats_features_upper


from Read_In_Data.read_data import Reading_Data
from Feature_Generation.create_binary_labels import Create_Binary_Labels
from Feature_Generation.summary_stats import Summary_Stats_Features
from Feature_Generation.difference_features import Difference_Features

import os

# Data Location
reading_data=Reading_Data()
# Create Binary y Variables
create_binary_labels=Create_Binary_Labels()
# Features Daily Stats Module
summary_stats_features=Summary_Stats_Features()
# Features Differences
difference_features=Difference_Features()


training_files_directory=os.listdir('/cephfs/interpolation/train')
training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]



# tester for now
training_files=training_files[0:1]
print(training_files)

for training_file in training_files:
    # Read in Data
    training_df=reading_data.read_in_pyspark_data_for_summary_stats(data_location=train_data_storage)
    
    training_df.show(2)