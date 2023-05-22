################################ Libraries ################################
from Read_In_Data.read_data import Reading_Data
import os


################################ Read In Modules ################################
reading_data=Reading_Data()


################################ Read In Data ################################
# Cross Validation Summary Stats Data
val_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/val'), x),os.listdir('/cephfs/summary_stats/val')))
val_files=[i for i in val_files if not ('.crc' in i or 'SUCCESS' in i)]

read_data=Reading_Data()
summary_stats_val=read_data.read_in_all_summary_stats(file_list=val_files)
summary_stats_val.show(2)
print((summary_stats_val.count(), len(summary_stats_val.columns)))