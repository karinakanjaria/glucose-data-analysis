import yaml
import numpy as np
# import fathon
# from fathon import fathonUtils as fu

with open('Input_Variables/data_vars.yaml', 'r') as file:
    input_vars=yaml.safe_load(file)


################################### Data Location ###################################
train_data_storage=input_vars['Data_Storage']['train_data_storage']
validation_data_storage=input_vars['Data_Storage']['validation_data_storage']
test_data_storage=input_vars['Data_Storage']['test_data_storage']

inter_train_location=input_vars['Data_Storage']['inter_train_location']
inter_test_location=input_vars['Data_Storage']['inter_test_location']
inter_val_location=input_vars['Data_Storage']['inter_val_location']

one_hot_encoding_data=input_vars['Data_Storage']['one_hot_encoding_location']


################################### Evaluation Metrics ###################################
evaluation_metrics_output_storage=input_vars['Evaluation_Metrics']['evaluation_metrics_output_storage']



################################### Feature Importance ###################################
feature_importance_storage_location=input_vars['Feature_Importance']['feature_importance_storage_location']
overall_feature_importance_plot_location=input_vars['Feature_Importance']['overall_feature_importance_plot_location']

################################### Analysis ###################################
analysis_group=input_vars['Analysis']['Analysis_Group']


################################### Time Series Input Features ###################################
mfdfa_win_1=input_vars['Time_Series_Input_Features']['MFDA']['win_1']
mfdfa_win_2=input_vars['Time_Series_Input_Features']['MFDA']['win_2']
# mfda_win_sizes=fu.linRangeByStep(mfdfa_win_1, mfdfa_win_2) # 30 mins to 1/2 year for

mfdfa_q_list_1=input_vars['Time_Series_Input_Features']['MFDA']['q_list_1']
mfdfa_q_list_2=input_vars['Time_Series_Input_Features']['MFDA']['q_list_2']
mfda_q_list=np.arange(mfdfa_q_list_1, mfdfa_q_list_2)

mfdfa_rev_seg=input_vars['Time_Series_Input_Features']['MFDA']['rev_seg']
mfdfa_pol_order=input_vars['Time_Series_Input_Features']['MFDA']['pol_order']

################################### Daily Stats Features ###################################
daily_stats_features_lower=input_vars['Daily_Stats_Features']['lower']
daily_stats_features_upper=input_vars['Daily_Stats_Features']['upper']


################################### ML Models ###################################
xgb_reg_model_storage_location=input_vars['ML_Models']['xgb_reg_model_storage_location']
xgb_class_model_storage_location=input_vars['ML_Models']['xgb_class_model_storage_location']

random_seed=input_vars['ML_Models']['random_seed']