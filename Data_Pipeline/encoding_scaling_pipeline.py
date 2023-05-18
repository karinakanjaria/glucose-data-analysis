from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, LongType
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql import Window


import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 

# CUSTOM TRANSFORMER ----------------------------------------------------------------
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
    
        va2 = VectorAssembler(inputCols=featureArr, outputCol="features")

        stages=[columns_scaler]+[va2]
        
        return stages
    
    
    
#         double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
#         float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
#         long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

#         # all_numerical=list(set(double_cols+float_cols))
#         # all_numerical_lags=[x for x in all_numerical if "lag" in x]
#         all_numerical=list(set(double_cols+float_cols+long_cols))
#         all_numerical.remove('target')
        
#         # featureArr = [('scaled_' + f) for f in all_numerical_lags]
#         # featureArr = [('scaled_' + f) for f in all_numerical]+['Sex_Encoded', 'Treatment_Encoded', 'AgeGroup_Encoded']
#         featureArr = [('scaled_' + f) for f in all_numerical]

#         va1 = [VectorAssembler(inputCols=[f], outputCol=('vec_' + f)) for f in all_numerical]
#         ss = [StandardScaler(inputCol='vec_' + f, outputCol='scaled_' + f, withMean=True, withStd=True) for f in all_numerical]

#         va2 = VectorAssembler(inputCols=featureArr, outputCol="features")

#         stages = va1 + ss + [va2]
    
#         return stages



#         double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
#         float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
#         long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

#         all_numerical=list(set(double_cols+float_cols+long_cols))
#         all_numerical.remove('target')

#         featureArr = [('scaled_' + f) for f in all_numerical]
        
#         for num_column in all_numerical:
#             input_col = f"{num_column}"
#             output_col = f"scaled_{num_column}"

#             w = Window.partitionBy('NumId')

#             mu = mean(input_col).over(w)
#             sigma = stddev(input_col).over(w)

#             df=df.withColumn(output_col, (col(input_col) - mu)/(sigma))