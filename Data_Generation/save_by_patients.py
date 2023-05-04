import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import StructType, StructField, \
StringType, IntegerType, TimestampType, DateType, FloatType
import time
from pyspark.sql.functions import col, lag, when, isnull, lit
import pathlib
from pyspark.sql.functions import col, to_date
from pyspark.sql.window import Window

class SaveByPatients:
    def __init__(self):
        self.rawSchema = StructType([StructField('_c0', IntegerType(),True),
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
        
        self.cohortSchema = StructType([StructField('', IntegerType(), True),
                            StructField('UserId', StringType(), True),
                            StructField('Gender', StringType(), True),
                            StructField('DOB', TimestampType(), True),
                            StructField('Age', IntegerType(), True),
                            StructField('DiabetesType', StringType(), True),
                            StructField('Treatment', StringType(), True)
                        ])
        
        
    def save_by_patients(self):
        spark = Spark_Session().spark
        
        allPaths = [str(x) for x in list(pathlib.Path('/cephfs/data').glob('*.csv')) if 'glucose_records' in str(x)]
        allPaths.sort()
        
        cohortDf = spark.read.options(delimiter=',')\
                    .csv('/cephfs/data/cohort.csv', header=True, schema=self.cohortSchema)\
                    .withColumnRenamed('', 'NumId')

        for idx in range(len(allPaths)):
            path = allPaths[idx]

            df = spark.read\
                    .format('csv')\
                    .option('delimiter', ',')\
                    .option("mode", "DROPMALFORMED")\
                    .option("header", True)\
                    .schema(raw_schema)\
                    .load(path)\
                    .select(col("PatientId"), col("Value"), \
                            col("GlucoseDisplayTime"), col("GlucoseDisplayTimeRaw"))

            df = df.withColumn('GlucoseDisplayDate',
                                   to_date(col('GlucoseDisplayTime')))

            patientIds = df.select('PatientId').distinct().select(col('PatientId')).collect()

            for patIds in patientIds:
                numId = cohortDf.filter(cohortDf.UserId == patIds.PatientId)\
                    .first()\
                    .NumId
                
                #get patient data
                patData = df.filter(df.PatientId == patIds.PatientId)
                
                #fill in with difference feature
                #patData = add_difference_features(df)
                
                #add numId to df
                patData = patData.withColumn('Num_Id', lit(numId))
                
                #write to patient_data
                patData.repartition(1).write\
                            .mode('append')\
                            .parquet('/cephfs/patient_data/patient_' + str(numId)) 

        