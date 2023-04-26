import pyspark
from pyspark import pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import OneHotEncoder, StringIndexer

class OneHotEncoding:
    def add_encoding_to_patients(self)
        spark = Spark_Session().spark
        
        df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv('/cephfs/data/cohort.csv')
        
        # assign index to string vals for OneHotEncoding
        encodedCols = ['Gender', 'Treatment'] # not doing'DiabetesType' because all type-two
        encodedLabels = []

        for name in encodedCols:
            indexer = StringIndexer(inputCol=name, outputCol= name + '_Num')
            indexer_fitted = indexer.fit(df)
            encodedLabels.append([name, indexer_fitted.labels])

            df = indexer_fitted.transform(df)
            
        #if you want to understand what each encoding label means    
        # order of index is based on frequency, most freq at beginning
        #[['Gender', ['Female', 'Male']],
        #['Treatment', ['yes-both', 'yes-long-acting', 'no', 'yes-fast-acting']]]
        
        ohe_gender = OneHotEncoder(inputCol="Gender_Num", outputCol="Gender_Encoded", dropLast=True)
        df = ohe_gender.fit(df).transform(df)

        ohe_treatment = OneHotEncoder(inputCol="Treatment_Num", outputCol="Treatment_Encoded", dropLast=True)
        df = ohe_treatment.fit(df).transform(df)
        
        df.write.parquet('/cephfs/data/cohort_encoded.parquet')