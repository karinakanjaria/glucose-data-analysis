from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import numpy as np
import random


[
 'Mean',
 'Std Dev',
 'Median',
 'Min',
 'Max',
 'CountBelow',
 'CountAbove',
 'PercentageBelow',
 'PercentageAbove',
 'lag_1',
 'lag_2',
 'lag_3']


class Numerical_Scaling:
    def create_standard_scaling(self, df):
        cols_to_scale = ['c', 'd', 'e']
        cols_to_keep_unscaled = ['a', 'b']

        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
        assembler = VectorAssembler().setInputCols(cols_to_scale).setOutputCol("features")
        sdf_transformed = assembler.transform(sdf)
        scaler_model = scaler.fit(sdf_transformed.select("features"))
        sdf_scaled = scaler_model.transform(sdf_transformed)

        sdf_scaled.show()