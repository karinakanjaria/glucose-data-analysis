from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import DoubleType, FloatType
from pyspark.sql.functions import udf
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

        assembler = [VectorAssembler(inputCols=[col], outputCol=col+'_vec') for col in all_numerical_lags]
        scale = [StandardScaler(inputCol=col+'_vec', outputCol=col+'_scaled') for col in all_numerical_lags]

        pipe = Pipeline(stages = assembler + scale)
        df_scale = pipe.fit(df).transform(df)
        
#         scaled_vals=df_scale.columns
#         scaled_feats=[x for x in scaled_vals if "scaled" in x]
        
#         udf1 = udf(lambda x : int(x[0]),FloatType())
        
#         for num_feature in scaled_feats:
#             df_scale=df_scale.select(num_feature, udf1(num_feature))
        
        return df_scale