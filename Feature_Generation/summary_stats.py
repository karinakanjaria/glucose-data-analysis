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
                                   chunk_val = 288):  

        group_cols = ["PatientId", "Chunk"]

        summary_df = df.groupby(group_cols)\
            .agg(max('y_binary').alias('y_summary_binary'),\
                 avg("Value").alias("Mean"),\
                 stddev("Value").alias("Std Dev"),\
                 percentile_approx("Value", .5).alias("Median"), \
                 min("Value").alias("Min"),\
                 max("Value").alias("Max"),\
                 avg('FirstDiff').alias('AvgFirstDiff'),\
                 avg('SecDiff').alias('AvgSecDiff'),\
                 stddev('FirstDiff').alias('StdFirstDiff'),\
                 stddev('SecDiff').alias('StdSecDiff'),\
                 sum(col("is_above")).alias("CountAbove"),\
                 sum(col("is_below")).alias("CountBelow"),\
                 sum(col('y_Binary')).alias('target')
                )

        return summary_df