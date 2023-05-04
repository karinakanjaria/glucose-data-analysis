# Python Libraries
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import when, col, rank, monotonically_increasing_id, date_trunc
from pyspark.sql.types import StructType, StructField, TimestampType

# import warnings
# warnings.filterwarnings('ignore')
class Value_Imputation:
    '''preprocessing stuff'''
    def cleanup(self, df):
        ''' INPUT
            df:    a spark DataFrame with one patient
            OUTPUT
            df:    a spark DataFrame with one patient
        '''
        
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
        
        df.collect() #is this something that has to be don regularly or no?
        
        """ ============== CREATE MISSING TIMESTAMPS ============== """
        """ the very next line of code is also the last that can be """
        """ done to the entire df of patients rather than just 1 patient"""
        
        '''get rid of seconds'''
        df = df.withColumn('GlucoseDisplayTime', date_trunc('minute', df.GlucoseDisplayTime))
        
        '''get patient IDs (takes ~30 seconds to run this) and then just 1 patient'''
        # patIds = [i.PatientId for i in df.select('PatientId').distinct().collect()]
        patient_str = "UNrE+DKLY8WEQsGhgJWIhSBlMYSn9szanvBQuoGKSjg="
        sql_filter = "PatientId = '" + patient_str + "'"
        subset = df.filter(sql_filter)
        
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
        dt_df = spark.createDataFrame(data=datetime_list, schema=deptSchema)
        
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
            -GlucoseDisplayTimeRaw should be used for checking the dates here, but implementation will have to come later
            -GlucoseDisplayDate is not filled out at this time'''
        merged = merged.drop('GlucoseDisplayTimeRaw') #someday i'll have time to use this as the double-checker
        merged = merged.fillna(patient_str, subset='PatientId')
        
        """ ============== FILL IN MISSING VALUES ============== """
        # filler = subset.agg({'Value': 'median'}).collect()[0][0]
        filler = subset.agg({'Value': 'avg'}).collect()[0][0]

        filled_df = merged.fillna(filler, subset='Value')
        
        return filled_df
        
    
    def cleanup_old(self, subset: pd.DataFrame):
        '''replace 0s with NaN to save a two steps down the line'''
        subset.loc[:,'Value'] = subset['Value'].replace(0, np.nan)

        '''drop duplicate datetimes for each patient'''
        subset = subset.drop_duplicates(subset='GlucoseDisplayTime', keep="first")

        # ogNon0Values = subset['Value'].replace(0, np.nan).count()
        # print("starting with", ogNon0Values, "nonzero glucose measurements ('Value')")
        # print(subset.info())    

        '''Find Unrecorded 5-minute Sample'''
        patID = subset['PatientId'].iloc[0]
        
        '''set datetime as index in order for resampling method below to work'''
        subset = subset.set_index('GlucoseDisplayTime')
        
        '''fill in missing timestamps'''
        subset = subset.resample('5min').first()

        '''fix columns that *need* to be filled in'''
        subset['PatientId'] = subset['PatientId'].replace(np.nan, patID)


        '''Interpolate Missing Data'''
        # Description goes here
        # 
        """
        subset['Value'] = subset['Value'].interpolate(method='pchip')
        missing_vals = subset[subset['missing'] == 1].index
        for i in missing_vals:
            lower_t = i - timedelta(hours = 5)
            upper_t = i - timedelta(hours = 3)
            std_prev = np.sqrt(np.std(subset.loc[lower_t:upper_t, 'Value']))
            jiggle = std_prev*np.random.randn()
            subset.loc[i, 'Value'] = subset.loc[i, 'Value'] + jiggle
        """
        temp = subset['Value'].mean()
        # subset[[subset['Value'].isna(), 'Value']] = temp
        subset['Value'] = subset['Value'].fillna(temp)
        
        # print(subset.columns)
        
        # .loc[row_indexer,col_indexer] = value
        
        subset=subset.reset_index(drop=False)
        
        subset=subset[['GlucoseDisplayTime', 'PatientId', 'Value']]
        
        return subset
