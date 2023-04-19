# Python Libraries
import numpy as np
import pandas as pd

from pyspark.sql.functions import col, to_date, sum, avg, max, min, \
stddev, percentile_approx,\
pandas_udf, PandasUDFType, lit, udf, collect_list, sqrt, monotonically_increasing_id, map_from_entries,\
rank, dense_rank, count, when

from pyspark.sql.window import Window

# class Summary_Stats_Features:
#     def pyspark_summary_statistics(self, df, output_schema, lower, upper):
#         @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
#         def transform_features(data):
#             median_df = pd.DataFrame(columns=['PatientId', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'Date'])
#             dates = data.GlucoseDisplayDate.unique()
#             patients=data.PatientId.unique()
            
#             for patient in patients:
#                 for date in dates:
#                     date_sub = data.loc[(data.GlucoseDisplayDate == date) & (data.PatientId == patient)]
                    
#                     # obtain summary values
#                     median = np.nanmedian(date_sub.Value)
#                     mean = np.nanmean(date_sub.Value)
#                     std = np.nanstd(date_sub.Value)
#                     maxVal = np.nanmax(date_sub.Value)
#                     minVal = np.nanmin(date_sub.Value)
                    
#                     # obtain areas above and below recommendation
#                     upperSegs = date_sub.loc[date_sub.Value > upper].Value - upper
#                     areaAbove = np.nansum(upperSegs)
#                     lowerSegs = -(date_sub.loc[date_sub.Value < lower].Value - lower)
#                     areaBelow = np.nansum(lowerSegs)
                    
#                     sample = [date_sub.iloc[0]["PatientId"], median, mean, std, maxVal, minVal, areaBelow, areaAbove, date]
                    
#                     median_df.loc[len(median_df.index)] = sample
            
#             merged_median_df=data.merge(median_df, left_on=['PatientId', 'GlucoseDisplayDate'], right_on=['PatientId', 'Date'], how='left')
#             merged_median_df=merged_median_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate',
#                                                'inserted', 'missing', 'y_Binary', 'Median', 'Mean', 'Std Dev', 'Max', 'Min',
#                                                'AreaBelow', 'AreaAbove']]
#             return merged_median_df
        
#         df=df.withColumn('Group', lit(1))
#         added_daily_features=df.groupby('Group').apply(transform_features)
        
#         return added_daily_features

class Summary_Stats_Features:
    def create_partition_date(self, df, chunk_val):
        window = Window.partitionBy(df['PatientId']).orderBy(df['GlucoseDisplayTime'])
        df = df.select('*', rank().over(window).alias('index'))
        df = df.withColumn("Chunk", (df.index/chunk_val).cast(IntegerType()))

        return df


    def pyspark_summary_statistics(self, df, \
                                   daily_stats_features_lower,\
                                   daily_stats_features_upper, \
                                   chunk_val = 12):  
        
        df_added = create_partition_date(df, chunk_val)
        group_cols = ["PatientId", "Chunk"]

        summary_df = df_added.groupby(group_cols)\
            .agg(avg("Value").alias("Mean"),\
                 stddev("Value").alias("Std Dev"),\
                 percentile_approx("Value", .5).alias("Median"), \
                 min("Value").alias("Min"),\
                 max("Value").alias("Max"),\
                 count(when(col("Value") < daily_stats_features_lower, 1)).alias("CountBelow"),\
                 count(when(col("Value") > daily_stats_features_upper, 1)).alias("CountAbove"),\
                 (count(when(col("Value") < daily_stats_features_lower, 1))/chunk_val).alias("PercentageBelow"),\
                 (count(when(col("Value") > daily_stats_features_upper, 1))/chunk_val).alias("PercentageAbove")
                )

        return summary_df