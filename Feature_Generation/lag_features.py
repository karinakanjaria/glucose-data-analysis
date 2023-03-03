import pyspark.sql.functions as F
from pyspark.sql.window import Window

class Create_Lagged_Features:
    def pyspark_lag_features(self, df, time_series_lag_values_created):
        w=Window.partitionBy('PatientId').orderBy('GlucoseDisplayTime')
        max_time_lag=time_series_lag_values_created+1

        for i in range(1, max_time_lag): 
            df=df.withColumn(f"lag_{i}", F.lag(F.col('Value'), i).over(w))

        df=df.na.drop()

        return df