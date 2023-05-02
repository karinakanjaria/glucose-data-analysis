# Python Libraries
import numpy as np
import pandas as pd

from pyspark.sql.functions import col, to_date, sum, avg, max, min, \
stddev, percentile_approx,\
pandas_udf, PandasUDFType, lit, udf, collect_list, sqrt, monotonically_increasing_id, map_from_entries,\
rank, dense_rank, count, when

from pyspark.sql.types import IntegerType

from pyspark.sql.window import Window

class Summary_Stats_Features:
    def create_chunk_col(self, df, chunk_val):
        window = Window.partitionBy(df['PatientId']).orderBy(df['GlucoseDisplayTime'])
        df = df.select('*', rank().over(window).alias('index'))
        df = df.withColumn("Chunk", (df.index/chunk_val).cast(IntegerType()))

        return df


    def pyspark_summary_statistics(self,
                                   df, \
                                   daily_stats_features_lower,\
                                   daily_stats_features_upper, \
                                   chunk_val = 12):  

        group_cols = ["PatientId", "Chunk"]

        summary_df = df_added.groupby(group_cols)\
            .agg(max('y_binary').alias('y_summary_binary'),\
                 avg("Value").alias("Mean"),\
                 stddev("Value").alias("Std Dev"),\
                 percentile_approx("Value", .5).alias("Median"), \
                 min("Value").alias("Min"),\
                 max("Value").alias("Max"),\
                 count(when(col("Value") < daily_stats_features_lower, 1)).alias("CountBelow"),\
                 count(when(col("Value") > daily_stats_features_upper, 1)).alias("CountAbove"),\
                 (count(when(col("Value") < daily_stats_features_lower, 1))/chunk_val).alias("PercentageBelow"),\
                 (count(when(col("Value") > daily_stats_features_upper, 1))/chunk_val).alias("PercentageAbove")
                )

        df_added = df_added.join(summary_df, ['PatientId', 'Chunk'])

        return df_added