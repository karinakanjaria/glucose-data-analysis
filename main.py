from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.sklearn_pipeline import Sklearn_Pipeline
from Data_Schema.schema import Pandas_UDF_Data_Schema

################################### PySpark UDF Schema Activation ###################################
pandas_udf_data_schema=Pandas_UDF_Data_Schema()


################################### Reading In Data Files in PySpark and Pandas ###################################
reading_data=Reading_Data(data_location='../capstone_data/ahr414_glucose_sample - ahr414_glucose_sample.csv')

####### PySpark
pyspark_df=reading_data.read_in_pyspark()
# print(pyspark_df.head(1))

####### Pandas
pandas_df=reading_data.read_in_pandas()
# print(pandas_df.head(1))



################################### Sklearn Pipeline in PySpark and Pure Sklearn ###################################
pandas_sklearn_pipeline=Sklearn_Pipeline()

####### PySpark
pyspark_transform_schema=pandas_udf_data_schema.sklearn_pyspark_schema()
pyspark_transformations=pandas_sklearn_pipeline.pyspark_sklearn_pipeline(df=pyspark_df, output_schema=pyspark_transform_schema)
# print(pyspark_transformations.head(1))

####### Pandas
pandas_transformations=pandas_sklearn_pipeline.pandas_transform_features(df=pandas_df)
# print(pandas_transformations.head(1))