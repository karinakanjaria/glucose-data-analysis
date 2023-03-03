import yaml
import numpy as np
# import fathon
# from fathon import fathonUtils as fu

with open('Input_Variables/data_vars.yaml', 'r') as file:
    input_vars=yaml.safe_load(file)


################################### Data Location ###################################
raw_data_storage=input_vars['Data_Storage']['raw_data_storage']


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
ml_models_train_split=input_vars['ML_Models']['train_split']
ml_models_test_split=input_vars['ML_Models']['test_split']


################################### Time Series Lagged Features ###################################
time_series_lag_values_created=input_vars['Time_Series_Lagged_Features']['Time_Series_Lag_Values_Created']