from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import DoubleType, FloatType
from pyspark.ml import Pipeline

class Feature_Transformations:
    def categorical_encoding(self, df):
        # Need  Categroical Features
    
        return None
    
    def numerical_scaling(self, df):
        double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
        float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]

        all_numerical=list(set(double_cols+float_cols))
        all_numerical_lags=[x for x in all_numerical if "lag" in x]

        featureArr = [('scaled_' + f) for f in all_numerical_lags]

        va1 = [VectorAssembler(inputCols=[f], outputCol=('vec_' + f)) for f in all_numerical_lags]
        ss = [StandardScaler(inputCol='vec_' + f, outputCol='scaled_' + f, withMean=True, withStd=True) for f in all_numerical_lags]

        va2 = VectorAssembler(inputCols=featureArr, outputCol="features")

        stages = va1 + ss + [va2]

        p = Pipeline(stages=stages)
        fitted_num_df=p.fit(df).transform(df)

        return fitted_num_df