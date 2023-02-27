from pyspark.sql import functions as F

import numpy as np

class Create_Binary_Labels:
    def pyspark_binary_labels(self, df, lower, upper):
        df=df.withColumn('y_Binary', F.when(F.col('Value') > upper, 1)\
            .when(F.col('Value') < lower, 1)\
                .otherwise(0))

        return df

    def pandas_binary_labels(self, df, lower, upper):
        df['y_Binary']=np.where(((df['Value'] > upper)) | ((df['Value']<lower)), 1, 0)

        return df