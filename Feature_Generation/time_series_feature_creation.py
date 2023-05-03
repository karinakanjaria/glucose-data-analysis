import pandas as pd
import numpy as np
import EntropyHub as eH
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

class TS_Features:

    entropy_schema = StructType([StructField('PatientId', StringType(), True),
                             StructField('Chunk', IntegerType(), True),
                             StructField('Entropy', FloatType(), True)])

    @pandas_udf(entropy_schema, functionType=PandasUDFType.GROUPED_MAP)
    def entropy(self, df):
        patientid = df['PatientId'].iloc[0]
        chunk = df['Chunk'].iloc[0]

        entropy = eH.SampEn(df.Value.values, m=4)[0][-1]

        entropy_df = pd.DataFrame([[patientid] + [chunk] + [entropy]])
        entropy_df.columns=['PatientId', 'Chunk', 'Entropy']
        return entropy_df
    
    
    poincare_schema = StructType([StructField('PatientId', StringType(), True),
                              StructField('Chunk', IntegerType(), True),
                              StructField('ShortTermVariance', FloatType(), True),
                              StructField('LongTermVariance', FloatType(), True),
                              StructField('VarianceRatio', FloatType(), True)])
    
    @pandas_udf(poincare_schema, functionType=PandasUDFType.GROUPED_MAP)
    def poincare(self, df):
        patientid = df['PatientId'].iloc[0]
        chunk = df['Chunk'].iloc[0]
        glucose_differentials = np.diff(df.Value)

        st_dev_differentials = np.std(np.diff(glucose_differentials))
        st_dev_values = np.std(glucose_differentials)

        # measures the width of poincare cloud
        short_term_variation = round((1 / np.sqrt(2)) * st_dev_differentials, 3)

        # measures the length of the poincare cloud
        long_term_variation = round(np.sqrt((2 * st_dev_values ** 2) - (0.5 * st_dev_differentials ** 2)), 3)

        ratio = round(short_term_variation / long_term_variation, 3)
        poincare_df = pd.DataFrame([[patientid] + [chunk] + [short_term_variation] + [long_term_variation] + [ratio]])
        poincare_df.columns=['PatientId', 'Chunk', 'ShortTermVariance', 'LongTermVariance', 'VarianceRatio']
        return poincare_df
