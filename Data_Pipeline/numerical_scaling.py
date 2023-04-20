from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

class Numerical_Scaling:
    def create_standard_scaling(self, df):
        stage_string_index = [StringIndexer(inputCol=col, outputCol=col+' string_indexed') for col in cat_cols]
        stage_onehot_enc =   [OneHotEncoder(inputCol=col+' string_indexed', outputCol=col+' onehot_enc') for col in cat_cols]
        
        cat_cols = []
        
        
        
        cols_to_scale = ['c', 'd', 'e']
        cols_to_keep_unscaled = ['a', 'b']

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        assembler = VectorAssembler().setInputCols(cols_to_scale).setOutputCol("features")
        sdf_transformed = assembler.transform(sdf)
        scaler_model = scaler.fit(sdf_transformed.select("features"))
        sdf_scaled = scaler_model.transform(sdf_transformed)

        sdf_scaled.show()