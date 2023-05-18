from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, LongType
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql import Window


import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 

class ColumnScaler(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def _transform(self, df):
        double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
        float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
        long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

        all_numerical=list(set(double_cols+float_cols+long_cols))
        all_numerical.remove('target')
        
        for num_column in all_numerical:
            input_col = f"{num_column}"
            output_col = f"scaled_{num_column}"

            w = Window.partitionBy('NumId')

            mu = mean(input_col).over(w)
            sigma = stddev(input_col).over(w)

            df=df.withColumn(output_col, (col(input_col) - mu)/(sigma))
            
        return df


class Feature_Transformations:    
    def numerical_scaling(self, df):
        double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
        float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
        long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

        all_numerical=list(set(double_cols+float_cols+long_cols))
        all_numerical.remove('target')

        featureArr = [('scaled_' + f) for f in all_numerical]

        columns_scaler=ColumnScaler()
    
        va2 = VectorAssembler(inputCols=featureArr, outputCol="features", handleInvalid='skip')

        stages=[columns_scaler]+[va2]
        
        return stages