import numpy as np

from Input_Variables.read_vars import raw_data_storage, \
                                      mfdfa_wins, mfdfa_q_list, mfdfa_rev_seg, mfdfa_pol_order

from Data_Schema.schema import Pandas_UDF_Data_Schema
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.sklearn_pipeline import Sklearn_Pipeline
from Time_Series_Features.time_series_feature_creation import TS_Feature_Creation

################################### PySpark UDF Schema Activation ###################################
pandas_udf_data_schema=Pandas_UDF_Data_Schema()



################################### Reading In Data Files in PySpark and Pandas ###################################
reading_data=Reading_Data(data_location=raw_data_storage)

####### PySpark
pyspark_df=reading_data.read_in_pyspark()
print(pyspark_df.head(1))

####### Pandas
pandas_df=reading_data.read_in_pandas()
print(pandas_df.head(1))



################################### Sklearn Pipeline in PySpark and Pure Sklearn ###################################
pandas_sklearn_pipeline=Sklearn_Pipeline()

####### PySpark
pyspark_transform_schema=pandas_udf_data_schema.sklearn_pyspark_schema()
pyspark_transformations=pandas_sklearn_pipeline.pyspark_sklearn_pipeline(df=pyspark_df, 
                                                                         output_schema=pyspark_transform_schema)
print(pyspark_transformations.head(1))

####### Pandas
pandas_transformations=pandas_sklearn_pipeline.pandas_transform_features(df=pandas_df)
print(pandas_transformations.head(1))



################################### Multifractal Detrended Fluctuation Analysis, Poincare Analysis, Functional Principal Component Analysis, Entropy  ###################################
ts_feature_creation=TS_Feature_Creation()

# # Issue with fcpaWrapper --> Need to make it loop through every PatientID, also fathon library not working on Mac M1 or M2 chips according to documentation
# mfdfa=ts_feature_creation.do_mfdfa(data=pandas_df.Value.to_numpy(), 
#                                    win_sizes=mfdfa_wins, 
#                                    q_list=mfdfa_q_list, 
#                                    rev_seg=mfdfa_rev_seg, 
#                                    pol_order=mfdfa_pol_order)

# Function Works
poincare=ts_feature_creation.poincare_wrapper(data=pandas_df)
print(poincare)

# # Issue with fcpaWrapper --> Need to make it loop through every PatientID and how to define minTime variable
# functional_principal_component_analysis=ts_feature_creation.fpcaWrapper(rawData=pandas_df)

# Fuction Works
entropy=ts_feature_creation.entropy_calculation(data=pandas_df)
print(entropy)