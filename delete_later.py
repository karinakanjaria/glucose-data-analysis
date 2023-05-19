import os

training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/train'), x),os.listdir('/cephfs/summary_stats/train')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]
# training_files.remove('/cephfs/summary_stats/train/summary_stats_parquet_145_26.parquet')
print(len(training_files))
print(training_files)

# print(training_files.index('/cephfs/summary_stats/train/summary_stats_parquet_13_51.parquet'))






# from pathlib import *
# # files = (x for x in Path("/cephfs/summary_stats/train/") if x.is_file())

# files=[x for x in Path("/cephfs/summary_stats/train/").iterdir() if x.is_file()]
# print(files)
# print(len(files))