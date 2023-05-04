import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pyspark.sql.functions import pandas_udf, PandasUDFType
# from Data_Pipeline.fill_missing_data import Value_Imputation

from datetime import date, datetime, timedelta
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import when, col, rank, monotonically_increasing_id, date_trunc
from pyspark.sql.types import StructType, StructField, TimestampType

class Date_And_Value_Imputation:
    def __init__(self):
        # self.value_imputation=Value_Imputation()

        # copypasted from Read_In_Data/read_data.py
        self.spark = SparkSession.builder.appName("Glucose").getOrCreate()

    def pyspark_custom_imputation_pipeline(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                       FunctionTransformer(self.value_imputation.cleanup))])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data1)

            return transformed_data_df

        transformed_data=df.groupby(analysis_group).apply(transform_features)

        return transformed_data
    
    
    
    '''preprocessing stuff'''
    def cleanup(self, df):
        """ INPUT
            df:    spark DataFrame with all patients
            OUTPUT
            df:    spark DataFrame with all patients
        """
        
        '''getting rid of any dates from before the actual start-date of Feb 1, 2022'''
        df = df.filter("GlucoseDisplayDate > date'2022-01-31'")
        
        '''replace 0s with NaN and dropna (we don't want to mistakenly start or end with a NaN in the resampling steps)'''
        df = df.withColumn("Value", \
                           when(col("Value")=="0", None) \
                           .otherwise(col("Value")))
        df = df.na.drop(subset=['PatientId','Value','GlucoseDisplayTime'])
        
        '''drop duplicate datetimes for each patient (this takes ~30 sec per 10 days x 8000 patients)'''
        window = Window.partitionBy('GlucoseDisplayTime','PatientId').orderBy('tiebreak')
        df = (df
         .withColumn('tiebreak', monotonically_increasing_id())
         .withColumn('rank', rank().over(window))
         .filter(col('rank') == 1).drop('rank','tiebreak')
        )
        
        '''get rid of seconds'''
        df = df.withColumn('GlucoseDisplayTime', date_trunc('minute', df.GlucoseDisplayTime))
        
        return df
        
        
        
    def replace_missing(self, subset):
        """ INPUT
            subset:     spark DataFrame with 1 patient
            OUTPUT
            filled_df:  spark DataFrame with 1 patient (-1 columns) and all missing rows filled in; not sorted
        """
        
        '''get first and last date (takes about 10 seconds per ten days of one patient)'''
        minimum = subset.agg({'GlucoseDisplayTime': 'min'}).collect()[0][0]
        maximum = subset.agg({'GlucoseDisplayTime': 'max'}).collect()[0][0]
        
        '''make a range that fills all those in'''
        def date_range_list(start_date, end_date):
            if start_date > end_date:
                raise ValueError("start_date must come before end_date")

            datetime_list = []
            curr_date = start_date
            while curr_date <= end_date:
                datetime_list.append([curr_date])
                curr_date += timedelta(minutes=5)
            return datetime_list
        
        datetime_list = date_range_list(minimum, maximum)
        
        '''make a dataframe of those dates'''
        deptSchema = StructType([       
            StructField('GlucoseDisplayTime', TimestampType(), True)
        ])
        dt_df = self.spark.createDataFrame(data=datetime_list, schema=deptSchema)
        
        '''merge og dataframe back into the new one'''
        merged = subset.unionByName(dt_df, allowMissingColumns=True)
        
        '''get rid of the timestamps we already have (using the exact same method as from "drop duplicate datetimes for each patient" above)'''
        window = Window.partitionBy('GlucoseDisplayTime').orderBy('tiebreak')
        merged = (merged
         .withColumn('tiebreak', monotonically_increasing_id())
         .withColumn('rank', rank().over(window))
         .filter(col('rank') == 1).drop('rank','tiebreak')
        )
        
        '''filling out the columns as needed:
            -PatientId should be all the same string
            -GlucoseDisplayTimeRaw should be used for checking the dates here, but implementation will have to come later'''
        merged = merged.fillna(patient_str, subset='PatientId')
        merged = merged.drop('GlucoseDisplayTimeRaw') #someday i'll have time to use this as the double-checker
        merged = merged.withColumn('GlucoseDisplayDate',
                                   to_date(col('GlucoseDisplayTime')))
        
        """ ============== FILL IN MISSING VALUES ============== """
        # filler = subset.agg({'Value': 'median'}).collect()[0][0]
        filler = subset.agg({'Value': 'avg'}).collect()[0][0]

        filled_df = merged.fillna(filler, subset='Value')
        
        return filled_df
    
    
    
    def impute_data(self, df):
        df = self.cleanup(df)
        
        patientIds = [i.NumId for i in df.select('NumId').distinct().collect()]
        
        for ids in patietnIds:
            # newData = #filter out only patient ids
            subset = df.filter("NumId = '" + ids + "'")
            
            # apply funciton to get a df of the new data
            subset = self.replace_missing(subset)
            
            # append that to the original df
            try:
                df_new = df_new.union(subset)
            except:
                try:
                    df_new = subset
                except:
                    raise ValueError("subsets of patient dataframes unable to append")
            
            # OR do a group by and an agg fxn
            # to do a custom agg fxn 
            