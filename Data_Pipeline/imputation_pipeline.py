import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pyspark.sql.functions import pandas_udf, PandasUDFType
from Data_Pipeline.fill_missing_data import Value_Imputation

from datetime import date, datetime, timedelta
import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import when, col, rank, monotonically_increasing_id, date_trunc, udf, to_date
from pyspark.sql.types import StructType, StructField, TimestampType

class Date_And_Value_Imputation:
    def __init__(self, spark):
        self.value_imputation=Value_Imputation()
        self.spark = SparkSession.builder.appName("Glucose").getOrCreate()
        self.inter_schema = StructType([StructField('NumId', IntegerType(), True),
                                        StructField('GlucoseDisplayTime', TimestampType(), True),
                                        StructField('Value', IntegerType(), True)])

        # # copypasted from Read_In_Data/read_data.py
        # self.spark = SparkSession.builder.appName("Glucose").getOrCreate()

        
    @pandas_udf(merged_schema, PandasUDFType.GROUPED_MAP)
    def interpolation(self, df):    
        min_max = test_df.groupby('NumId')\
                    .agg({'GlucoseDisplayTime' : ['min', 'max']})

        merge_df = pd.DataFrame(columns=['GlucoseDisplayTime', 'NumId'])
        for idx, row in min_max.iterrows():
            #grab all poteitnal dates in range

            date_df = pd.DataFrame(pd.date_range(row[0], row[1], freq='5min'), columns=['GlucoseDisplayTime'])                              
            date_df['NumId']= idx

            # merge dates with big pypsark df
            merged = test_df[test_df['NumId'] == idx]\
                    .merge(date_df, how='right', on=['GlucoseDisplayTime', 'NumId'])\
                    .sort_values(by=['GlucoseDisplayTime', 'Value'], na_position='last')

            merged['TimeLag'] = np.concatenate((merged['GlucoseDisplayTime'].iloc[1:].values,\
                                                np.array(merged['GlucoseDisplayTime'].iloc[-1])), axis=None)\
                                .astype('datetime64[ns]')

            merged['Diff'] = (merged['TimeLag'] - merged['GlucoseDisplayTime']).dt.seconds

            len_merged = len(merged)

            # get all index of rows with diff less than 5 mins, add 1 to remove next row, 
            # dont include last row to delete
            indexes_to_remove = [x for x in merged[merged['Diff'] < 300].index + 1 if x < len_merged]

            if len(indexes_to_remove) > 0:
                merged = merged.drop(indexes_to_remove)

            # its ready freddy for some interpoletty
            # grab all potential dates in range

            date_df = pd.DataFrame(pd.date_range(row[0], row[1], freq='5min'), columns=['GlucoseDisplayTime'])                              
            date_df['NumId']= idx

            # merge dates with big pypsark df
            merged = test_df[test_df['NumId'] == idx]\
                    .merge(date_df, how='right', on=['GlucoseDisplayTime', 'NumId'])\
                    .sort_values(by=['GlucoseDisplayTime', 'Value'], na_position='last')

            merged['TimeLag'] = np.concatenate((merged['GlucoseDisplayTime'].iloc[1:].values,\
                                                np.array(merged['GlucoseDisplayTime'].iloc[-1])), axis=None)\
                                .astype('datetime64[ns]')

            merged['Diff'] = (merged['TimeLag'] - merged['GlucoseDisplayTime']).dt.seconds

            len_merged = len(merged)

            # get all index of rows with diff less than 5 mins, add 1 to remove next row, 
            # dont include last row to delete
            indexes_to_remove = [x for x in merged[merged['Diff'] < 300].index + 1 if x < len_merged]

            if len(indexes_to_remove) > 0:
                merged = merged.drop(indexes_to_remove)

            # its ready freddy for some interpoletty
            # merged DF is the dataframe ready to go into interpolation function


    
    def pyspark_custom_imputation_pipeline(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                       FunctionTransformer(self.value_imputation.cleanup_old))])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data1)

            return transformed_data_df

        transformed_data=df.groupby(analysis_group).apply(transform_features)

        return transformed_data        

    
#     '''preprocessing stuff'''
#     def cleanup(self, df):
#         '''get rid of seconds'''
#         df = df.withColumn('GlucoseDisplayTime', date_trunc('minute', df.GlucoseDisplayTime))
        
#         return df
        
        
        
    def replace_missing(self, subset, patient_str):
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
        # df = self.cleanup(df)
        
        patientIds = [i.NumId for i in df.select('NumId').distinct().collect()]
        
        for ids in patientIds:
            # newData = #filter out only patient ids
            subset = df.filter("NumId = '" + str(ids) + "'")
            
            # apply funciton to get a df of the new data
            subset = self.replace_missing(subset,ids)
            
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
        
        self.spark.stop()
        
        return df_new