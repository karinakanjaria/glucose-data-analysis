import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, LongType
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql import Window

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 

from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/train'), x),os.listdir('/cephfs/summary_stats/train')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]



class Reading_Data:
    def __init__(self):
        self.spark = SparkSession.builder.appName("Glucose").getOrCreate()
        # self.data_location='/cephfs/summary_stats/train'
        # self.data_location='/cephfs/summary_stats/train/summary_stats_parquet_0_25.parquet'
        
       
        # self.glucose_data_schema=StructType([StructField('NumId', IntegerType(), True), 
        #                                      StructField('Chunk', IntegerType(), True), 
        #                                      StructField('Mean', DoubleType(), True), 
        #                                      StructField('StdDev', DoubleType(), True), 
        #                                      StructField('Median', FloatType(), True), 
        #                                      StructField('Min', FloatType(), True), 
        #                                      StructField('Max', FloatType(), True), 
        #                                      StructField('AvgFirstDiff', DoubleType(), True), 
        #                                      StructField('AvgSecDiff', DoubleType(), True), 
        #                                      StructField('StdFirstDiff', DoubleType(), True), 
        #                                      StructField('StdSecDiff', DoubleType(), True), 
        #                                      StructField('CountAbove', LongType(), True), 
        #                                      StructField('CountBelow', LongType(), True), 
        #                                      StructField('TotalOutOfRange', LongType(), True), 
        #                                      StructField('target', LongType(), True)])
        
        

    # def read_in_all_summary_stats(self):
    #     summary_stats_df = self.spark.read \
    #                            .schema(self.glucose_data_schema) \
    #                            .format('parquet') \
    #                            .load(self.data_location)
        
        # summary_stats_df = self.spark.read \
        #                        .format('parquet') \
        #                        .load(self.data_location)

    # def read_in_all_summary_stats(self, data_location):        
    #     summary_stats_df = self.spark.read \
    #                            .format('parquet') \
    #                            .load(data_location)
    
    def read_in_all_summary_stats(self):        
        summary_stats_df = self.spark.read \
                               .parquet(*training_files)



        return summary_stats_df

read_data=Reading_Data()
summary_stats_df=read_data.read_in_all_summary_stats()
print((summary_stats_df.count(), len(summary_stats_df.columns)))

from pyspark.sql.functions import countDistinct
df2=summary_stats_df.select(countDistinct("NumId"))
df2.show()


# training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/train'), x),os.listdir('/cephfs/summary_stats/train')))
# training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]
# # # # training_files.remove('/cephfs/summary_stats/train/summary_stats_parquet_145_26.parquet')
# number_of_files=len(training_files)
# counter = 1

# empty_list=[]

# read_data=Reading_Data()
# for file in training_files:
#     print(f'Starting {counter}/{number_of_files} : {file}')
#     summary_stats_df=read_data.read_in_all_summary_stats(data_location=file)
#     summary_stats_df.show(1)
#     counter=counter+1
#     if summary_stats_df.rdd.isEmpty():
#         empty_list.append(file)
#         continue
#     else:
#         continue
# print(empty_list)

# read_data=Reading_Data()
# for file in training_files:
#     print(f'Completed: {file}')
#     summary_stats_df=read_data.read_in_all_summary_stats(data_location=file)
#     summary_stats_df.show(1)
    

    

    
# class ColumnScaler(Transformer, DefaultParamsReadable, DefaultParamsWritable):
#     def _transform(self, df):
#         double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
#         float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
#         long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

#         all_numerical=list(set(double_cols+float_cols+long_cols))
#         all_numerical.remove('target')
        
#         for num_column in all_numerical:
#             input_col = f"{num_column}"
#             output_col = f"scaled_{num_column}"

#             w = Window.partitionBy('NumId')

#             mu = mean(input_col).over(w)
#             sigma = stddev(input_col).over(w)

#             df=df.withColumn(output_col, (col(input_col) - mu)/(sigma))
            
#         return df


# class Feature_Transformations:    
#     def numerical_scaling(self, df):
#         double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
#         float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
#         long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

#         all_numerical=list(set(double_cols+float_cols+long_cols))
#         all_numerical.remove('target')

#         featureArr = [('scaled_' + f) for f in all_numerical]

#         columns_scaler=ColumnScaler()
    
#         va2 = VectorAssembler(inputCols=featureArr, outputCol="features", handleInvalid='skip')

#         stages=[columns_scaler]+[va2]
        
#         return stages
    
# feature_transformations=Feature_Transformations()
# pipeline_transformation_stages=feature_transformations.numerical_scaling(df=summary_stats_df)





# class Create_PySpark_XGBoost:
#     def __init__(self):
#         self.features_col="features"
#         self.label_name="target"
        
    
#     def initial_training_xgboost_regression(self, ml_df, stages, random_seed):
#         # xgb_regression=SparkXGBRegressor(features_col=features_col, 
#         #                                   label_col=label_name,
#         #                                   num_workers=4,
#         #                                   random_state=random_seed,
#         #                                   use_gpu=True)
        
#         initial_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
#                                                  label_col=self.label_name,
#                                                  random_state=random_seed,
#                                                  use_gpu=False)

#         stages.append(initial_xgb_regression)
#         pipeline=Pipeline(stages=stages)
        
#         model=pipeline.fit(ml_df)
        
#         return model
    
# create_pyspark_xgboost=Create_PySpark_XGBoost()
# xgboost_regression_model=create_pyspark_xgboost.initial_training_xgboost_regression(ml_df=summary_stats_df, 
#                                                                                     stages=pipeline_transformation_stages, 
#                                                                                     random_seed=123)

# model_storage_location='/cephfs/Saved_Models/Summary_Stats_Model'
# xgboost_regression_model.write().overwrite().save(model_storage_location)