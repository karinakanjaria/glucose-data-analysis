# load in imports
#!sudo apt-get update
#!sudo apt-get install openjdk-8-jdk
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DateType, FloatType
import time
import pathlib
from pyspark.sql.functions import col, to_date

### Step 0. Data Generation 
'''
    1. Reads in raw csv files
    2. Select PatientId, Value, GlucoseDisplayTime
    3. Generate GlucoseDisplayDate
    4. Save to "../../cephfs/filterParquet/reduced_data_days_0_to_9.parquet" (10 day chunks)
    
    Parquet File Schema
     |-- PatientId: string (nullable = true)
     |-- Value: integer (nullable = true)
     |-- GlucoseDisplayTime: timestamp (nullable = true)
     |-- GlucoseDisplayTimeRaw: timestamp (nullable = true)
     |-- GlucoseDisplayDate: date (nullable = true)
'''

class Spark_Session:
    def __init__(self):
        self.conf = pyspark.SparkConf().setAll([\
            ('spark.app.name', 'Glucose_Analysis_Spark')])
        self.spark = SparkSession.builder.config(conf=self.conf)\
            .getOrCreate()        

class Create_Parquet_Files:
    
    def __init__(self):
        self.raw_data_schema = StructType([StructField('_c0', IntegerType(),True),
                                StructField('PostDate', TimestampType(),True),
                                StructField('IngestionDate', TimestampType(),True),
                                StructField('PostId', StringType(),True),
                                StructField('PostTime', TimestampType(), True),
                                StructField('PatientId', StringType(), True),
                                StructField('Stream', StringType(), True),
                                StructField('SequenceNumber', StringType(), True),
                                StructField('TransmitterNumber', StringType(), True),
                                StructField('ReceiverNumber', StringType(), True),
                                StructField('RecordedSystemTime', TimestampType(), True),
                                StructField('RecordedDisplayTime', TimestampType(), True),
                                StructField('RecordedDisplayTimeRaw', TimestampType(), True),
                                StructField('TransmitterId', StringType(), True),
                                StructField('TransmitterTime', StringType(), True),
                                StructField('GlucoseSystemTime', TimestampType(), True),
                                StructField('GlucoseDisplayTime', TimestampType(), True),
                                StructField('GlucoseDisplayTimeRaw', StringType(), True),
                                StructField('Value', FloatType(), True),
                                StructField('Status', StringType(), True),
                                StructField('TrendArrow', StringType(), True),
                                StructField('TrendRate', FloatType(), True),
                                StructField('IsBackFilled', StringType(), True),
                                StructField('InternalStatus', StringType(), True),
                                StructField('SessionStartTime', StringType(), True)])
    
    def create_parquet(self):
        spark = Spark_Session().spark
        
        # get all csv paths of raw data
        allPaths = [str(x) for x in list(pathlib.Path('/cephfs/data').glob('*.csv'))\
                    if 'glucose_records' in str(x)]

        # create date range to iterate over
        pathRange = spark.sparkContext.range(10, len(allPaths), 10).collect()
        pathRange.append(366) 

        # load in data, select columns to save, create date column, save to parquet format
        prevIdx = 0
        for idx in pathRange:
            paths = allPaths[prevIdx:idx]

            df = spark.read\
            .format('csv')\
            .option('delimiter', ',')\
            .option("mode", "DROPMALFORMED")\
            .option("header", True)\
            .schema(self.raw_data_schema)\
            .load(paths)\
            .select(col("PatientId"), col("Value"), \
                    col("GlucoseDisplayTime"), col("GlucoseDisplayTimeRaw"))
            
            df_toParq = df.withColumn('GlucoseDisplayDate',
                                   to_date(col('GlucoseDisplayTime')))

            df_toParq.repartition(1).write\
                .mode('overwrite')\
                .parquet('/cephfs/stepped_glucose_data/step0_load/parquet_' + str(prevIdx) + '_to_' + str(idx)) 

            prevIdx = idx    

            
        spark.stop()