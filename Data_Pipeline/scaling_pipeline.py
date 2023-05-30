from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Window
from pyspark.sql.functions import col, mean, stddev
from pyspark.ml.feature import VectorAssembler

class ColumnScaler(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def _transform(self, df):
        all_numerical=['ShortTermVariance', 'LongTermVariance', 'VarianceRatio', 'SampleEntropy', 
                       'PermutationEntropy', 'Mean', 'StdDev', 'Median', 'Min', 'Max', 'AvgFirstDiff', 'AvgSecDiff', 
                       'StdFirstDiff', 'StdSecDiff', 'CountAbove', 'CountBelow', 'TotalOutOfRange']
        
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
        all_numerical=['ShortTermVariance', 'LongTermVariance', 'VarianceRatio', 'SampleEntropy', 
                       'PermutationEntropy', 'Mean', 'StdDev', 'Median', 'Min', 'Max', 'AvgFirstDiff', 
                       'AvgSecDiff', 'StdFirstDiff', 'StdSecDiff', 'CountAbove', 'CountBelow', 'TotalOutOfRange']
        
        
        featureArr = [('scaled_' + f) for f in all_numerical]

        columns_scaler=ColumnScaler()
    
        va2 = VectorAssembler(inputCols=featureArr, outputCol="features", handleInvalid='skip')

        stages=[columns_scaler]+[va2]
        
        return stages