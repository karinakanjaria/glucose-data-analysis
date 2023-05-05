from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, LongType
from pyspark.ml import Pipeline

class Feature_Transformations:
    def categorical_encoding(self, df):    
        return None
    
    def numerical_scaling(self, df):
        double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
        float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
        long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

        # all_numerical=list(set(double_cols+float_cols))
        # all_numerical_lags=[x for x in all_numerical if "lag" in x]
        all_numerical=list(set(double_cols+float_cols+long_cols))
        all_numerical.remove('target')
        
        # featureArr = [('scaled_' + f) for f in all_numerical_lags]
        # featureArr = [('scaled_' + f) for f in all_numerical]+['Sex_Encoded', 'Treatment_Encoded', 'AgeGroup_Encoded']
        featureArr = [('scaled_' + f) for f in all_numerical]

        va1 = [VectorAssembler(inputCols=[f], outputCol=('vec_' + f)) for f in all_numerical]
        ss = [StandardScaler(inputCol='vec_' + f, outputCol='scaled_' + f, withMean=True, withStd=True) for f in all_numerical]

        va2 = VectorAssembler(inputCols=featureArr, outputCol="features")

        stages = va1 + ss + [va2]
    
        return stages