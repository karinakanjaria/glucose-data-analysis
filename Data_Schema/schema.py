from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, DateType, IntegerType, ArrayType

class Project_Data_Schema:
    def data_schema_pyspark(self):        
        glucose_data_schema=StructType([StructField('PostDate', TimestampType(),True),
                                        StructField('IngestionDate', TimestampType(),True),
                                        StructField('PostID', StringType(),True),
                                        StructField('PostTime', TimestampType(), True),
                                        StructField('PatientId', StringType(), True),
                                        StructField('Stram', StringType(), True),
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
                                        StructField('GlucoseDisplayTimeRaw', TimestampType(), True),
                                        StructField('Value', FloatType(), True),
                                        StructField('Status', StringType(), True),
                                        StructField('TrendArrow', StringType(), True),
                                        StructField('TrendRate', FloatType(), True),
                                        StructField('IsBackFilled', StringType(), True),
                                        StructField('InternalStatus', StringType(), True),
                                        StructField('SessionStartTime', StringType(), True)])
        return glucose_data_schema

    def data_schema_pandas(self):
        glucose_dtype = {'PostDate': str, 
                         'IngestionDate': str, 
                         'PostID': str, 
                         'PostTime': str, 
                         'PatientId': str, 
                         'Stram': str, 
                         'SequenceNumber': str, 
                         'TransmitterNumber': str, 
                         'ReceiverNumber': str, 
                         'RecordedSystemTime': str, 
                         'RecordedDisplayTime': str, 
                         'RecordedDisplayTimeRaw': str, 
                         'TransmitterId': str, 
                         'TransmitterTime': str, 
                         'GlucoseSystemTime': str, 
                         'GlucoseDisplayTime': str, 
                         'GlucoseDisplayTimeRaw': str, 
                         'Value': float, 
                         'Status': str, 
                         'TrendArrow': str, 
                         'TrendRate': float, 
                         'IsBackFilled': str, 
                         'InternalStatus': str, 
                         'SessionStartTime': str}

        parse_dates=['PostDate', 'IngestionDate', 'PostTime', 
                     'RecordedSystemTime', 'RecordedDisplayTime', 'RecordedDisplayTimeRaw', 
                     'GlucoseSystemTime', 'GlucoseDisplayTime', 'GlucoseDisplayTimeRaw']

        return glucose_dtype, parse_dates

class Pandas_UDF_Data_Schema:
    def custom_imputation_pyspark_schema(self):
        pyspark_custom_imputation_schema=StructType([StructField('PatientId', StringType(),True),
                                                     StructField('Value', FloatType(),True),
                                                     StructField('GlucoseDisplayTime', TimestampType(),True),
                                                     StructField('GlucoseDisplayDate', DateType(),True),
                                                     StructField('inserted', IntegerType(),True),
                                                     StructField('missing', IntegerType(),True)])

        return pyspark_custom_imputation_schema

    def summary_stats_schema(self):
        pyspark_summary_stats_schema=StructType([StructField('PatientId', StringType(),True),
                                                 StructField('Value', FloatType(),True),
                                                 StructField('GlucoseDisplayTime', TimestampType(),True),
                                                 StructField('GlucoseDisplayDate', DateType(),True),
                                                 StructField('inserted', IntegerType(),True),
                                                 StructField('missing', IntegerType(),True),
                                                 StructField('y_Binary', IntegerType(),True),
                                                 StructField('Median', FloatType(),True),
                                                 StructField('Mean', FloatType(),True),
                                                 StructField('Std Dev', FloatType(),True),
                                                 StructField('Max', FloatType(),True),
                                                 StructField('Min', FloatType(),True),
                                                 StructField('AreaBelow', FloatType(),True),
                                                 StructField('AreaAbove', FloatType(),True)])

        return pyspark_summary_stats_schema



    def sklearn_pyspark_categorical_schema(self):
        pyspark_categorical_schema=StructType([StructField('PatientId', StringType(),True),
                                               StructField('Value', FloatType(),True),
                                               StructField('GlucoseDisplayTime', TimestampType(),True),
                                               StructField('GlucoseDisplayDate', DateType(),True),
                                               StructField('y_Binary', IntegerType(),True),
                                               StructField('Median', FloatType(),True),
                                               StructField('Mean', FloatType(),True),
                                               StructField('Std Dev', FloatType(),True),
                                               StructField('Max', FloatType(),True),
                                               StructField('Min', FloatType(),True),
                                               StructField('AreaBelow', FloatType(),True),
                                               StructField('AreaAbove', FloatType(),True),
                                               StructField('inserted', ArrayType(IntegerType()),True),
                                               StructField('missing', ArrayType(IntegerType()),True)])

        return pyspark_categorical_schema



    def sklearn_pyspark_numerical_schema(self):
        pyspark_numerical_schema=StructType([StructField('PatientId', StringType(),True),
                                               StructField('Value', FloatType(),True),
                                               StructField('GlucoseDisplayTime', TimestampType(),True),
                                               StructField('GlucoseDisplayDate', DateType(),True),
                                               StructField('Median', FloatType(),True),
                                               StructField('Mean', FloatType(),True),
                                               StructField('Std Dev', FloatType(),True),
                                               StructField('Max', FloatType(),True),
                                               StructField('Min', FloatType(),True),
                                               StructField('AreaBelow', FloatType(),True),
                                               StructField('AreaAbove', FloatType(),True),
                                               StructField('inserted', ArrayType(IntegerType()),True),
                                               StructField('missing', ArrayType(IntegerType()),True),
                                               StructField('y_Binary', IntegerType(),True)])

        return pyspark_numerical_schema




    def XGBoost_schema(self):
        pyspark_xgboost_schema=StructType([StructField('PatientId', StringType(),True),
                                           StructField('GlucoseDisplayTime', TimestampType(), True),
                                           StructField('Predictions', IntegerType(),True),
                                           StructField('Actual', IntegerType(),True)])

        return pyspark_xgboost_schema