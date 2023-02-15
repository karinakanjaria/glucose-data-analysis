import yaml
import numpy as np
# import fathon
# from fathon import fathonUtils as fu

with open('Input_Variables/data_vars.yaml', 'r') as file:
    input_vars=yaml.safe_load(file)


################################### Data Location --> Will be pointing at an S3 bucket later ###################################
raw_data_storage=input_vars['Data_Storage']['raw_data_storage']


################################### Time Series Input Features ###################################
mfdfa_win_1=input_vars['Time_Series_Input_Features']['MFDA']['win_1']
mfdfa_win_2=input_vars['Time_Series_Input_Features']['MFDA']['win_2']
# mfdfa_wins=fu.linRangeByStep(mfdfa_win_1, mfdfa_win_2) # 30 mins to 1/2 year for
mfdfa_wins=None

mfdfa_q_list_1=input_vars['Time_Series_Input_Features']['MFDA']['q_list_1']
mfdfa_q_list_2=input_vars['Time_Series_Input_Features']['MFDA']['q_list_2']
mfdfa_q_list=np.arange(mfdfa_q_list_1, mfdfa_q_list_2)

mfdfa_rev_seg=input_vars['Time_Series_Input_Features']['MFDA']['rev_seg']
mfdfa_pol_order=input_vars['Time_Series_Input_Features']['MFDA']['pol_order']