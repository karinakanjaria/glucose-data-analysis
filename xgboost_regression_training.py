print("import 0")
from Input_Variables.read_vars import random_seed
print("import 1")

from Read_In_Data.read_data import Reading_Data
print("import 2")
from Data_Pipeline.encoding_scaling_pipeline import Feature_Transformations
print("import 3")
from Model_Creation.pyspark_xgboost import Create_PySpark_XGBoost
print("import 4")

import os
print("import 5. aaaand done!")

reading_data=Reading_Data()
feature_transformations=Feature_Transformations()
create_pyspark_xgboost=Create_PySpark_XGBoost()

# training_files_directory=os.listdir('/cephfs/summary_stats/train')
# training_files=[i for i in training_files_directory if not ('.crc' in i or 'SUCCESS' in i)]

training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/train'), x),os.listdir('/cephfs/summary_stats/train')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]
training_files.remove('/cephfs/summary_stats/train/summary_stats_parquet_145_26.parquet')

total_file_iteration=len(training_files)
iteration=1

print(total_file_iteration)

model_storage_location='/cephfs/Saved_Models/Summary_Stats_Model'

blank_interoplation_list=[]

for file in training_files:
    # Read in Summary Statistics
    summary_stats=reading_data \
        .read_in_pyspark_data_for_summary_stats(data_location=f'{file}')
    
    # Numerical Transformation Stages
    training_numerical_stages=feature_transformations.numerical_scaling(df=summary_stats)

    if iteration==1:
        # XGBoost Model
        xgboost_model=create_pyspark_xgboost \
            .initial_training_xgboost_regression(ml_df=summary_stats,
                                             stages=training_numerical_stages,
                                             random_seed=random_seed)
        
    
    else:
        try:
            xgboost_model=create_pyspark_xgboost \
                .batch_training_xgboost_regression(ml_df=summary_stats, 
                                                   stages=training_numerical_stages, 
                                                   random_seed=random_seed, 
                                                   past_model=xgboost_model)
            
            # if iteration%20 == 0 or iteration==total_file_iteration:
            #     xgboost_model.write().overwrite().save(model_storage_location)
            #     print('Saved Model')
            # else:
            #     None
        except:
            print(f'Failed {file}: {iteration}/{total_file_iteration}')
            blank_interoplation_list.append(f'{file}')
            iteration=iteration+1
            continue
            
    print(f'Completed {file}: {iteration}/{total_file_iteration}')
    iteration+=1
    
print(blank_interoplation_list)
xgboost_model.write().overwrite().save(model_storage_location)
print('Saved Model')