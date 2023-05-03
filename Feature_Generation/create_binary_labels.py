from pyspark.sql import functions as F

import numpy as np

class Create_Binary_Labels:    
    def __init__(self):
        self.lower = 70
        self.upper = 180
    
    def pyspark_binary_labels(self, df):
        #get 10th and 90th percentiles of patient
        lower_10, upper_90  = df.approxQuantile('Value', [.1, .9], 0)
        
        #if 10th percentile of patient is lower than default, use percentile
        lower = lower_10 if lower_10 < self.lower else self.lower
        upper = upper_90 if upper_90 > self.upper else self.upper
        
        df=df.withColumn('y_Binary', F.when(F.col('Value') > upper, 1)\
            .when(F.col('Value') < lower, 1)\
                .otherwise(0))
        df=df.withColumn('is_above', F.when(F.col('Value') > upper, 1).otherwise(0))
        df=df.withColumn('is_below', F.when(F.col('Value') < lower, 1).otherwise(0))

        return df

    def pandas_binary_labels(self, df, lower, upper):
        df['y_Binary']=np.where(((df['Value'] > upper)) | ((df['Value']<lower)), 1, 0)

        return df