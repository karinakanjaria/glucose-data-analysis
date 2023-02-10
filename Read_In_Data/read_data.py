# Python Libraries
import pandas as pd

# PySpark Libraries
from pyspark.sql import SparkSession

# Import Modules
from Data_Schema.schema import Project_Data_Schema

class Reading_Data:
    def __init__(self, data_location):    
        self.data_location=data_location
        self.project_data_schema=Project_Data_Schema()

        self.pyspark_data_schema=self.project_data_schema.data_schema_pyspark() 
        self.pandas_data_schema=self.project_data_schema.data_schema_pandas() 
             
    def read_in_pyspark(self):        
        spark=SparkSession.builder.master("local"). \
                           appName('Resd_Glucose_Data'). \
                           getOrCreate()
        
        pyspark_glucose_data=spark.read.csv(self.data_location, 
                                            header=True,
                                            sep=',',
                                            schema=self.pyspark_data_schema)
        
        return pyspark_glucose_data

    def read_in_pandas(self):                
        pandas_glucose_data=pd.read_csv(self.data_location,
                                        dtype=self.pandas_data_schema[0],
                                        parse_dates=self.pandas_data_schema[1])
        
        return pandas_glucose_data