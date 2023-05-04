# Python Libraries
import pandas as pd

# PySpark Libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_trunc, col, udf

# Import Modules
from Data_Schema.schema import Project_Data_Schema

class Reading_Data:
    def __init__(self):    
        self.project_data_schema=Project_Data_Schema()

        self.pyspark_data_schema=self.project_data_schema.data_schema_pyspark()
        self.spark= SparkSession.builder.appName("Glucose").getOrCreate()
            
            
    def read_in_pyspark_training(self, training_data_location):                 
        pyspark_glucose_data=self.spark.read.schema(self.pyspark_data_schema).parquet(training_data_location)
        pyspark_glucose_data=pyspark_glucose_data.withColumn("GlucoseDisplayTime", 
                                                             date_trunc("minute", 
                                                                        col("GlucoseDisplayTime")))
        pyspark_glucose_data=pyspark_glucose_data.distinct()
        
        pyspark_glucose_data=pyspark_glucose_data.orderBy("PatientId", "GlucoseDisplayTime",
                                                          ascending=True)
        
        return pyspark_glucose_data
    
    
    def read_in_pyspark_cross_validation(self, cross_validation_data_location):                
        pyspark_glucose_data=self.spark.read.schema(self.pyspark_data_schema).parquet(cross_validation_data_location)
        pyspark_glucose_data=pyspark_glucose_data.withColumn("GlucoseDisplayTime", 
                                                             date_trunc("minute", 
                                                                        col("GlucoseDisplayTime")))
        pyspark_glucose_data=pyspark_glucose_data.distinct()
        
        pyspark_glucose_data=pyspark_glucose_data.orderBy("PatientId", "GlucoseDisplayTime",
                                                          ascending=True)
        
        return pyspark_glucose_data
    
    
    
    def read_in_pyspark_testing(self, testing_data_location):
        pyspark_glucose_data=self.spark.read.schema(self.pyspark_data_schema).parquet(testing_data_location)
        pyspark_glucose_data=pyspark_glucose_data.withColumn("GlucoseDisplayTime", 
                                                             date_trunc("minute", 
                                                                        col("GlucoseDisplayTime")))
        pyspark_glucose_data=pyspark_glucose_data.distinct()
        
        pyspark_glucose_data=pyspark_glucose_data.orderBy("PatientId", "GlucoseDisplayTime",
                                                          ascending=True)
        
        return pyspark_glucose_data
    
    
    def read_in_one_hot_encoded_data(self, one_hot_encoding_location):
        one_hot_encoding_df=self.spark.read.parquet(one_hot_encoding_location)
        
        return one_hot_encoding_df
        