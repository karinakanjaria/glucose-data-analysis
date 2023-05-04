import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pyspark.sql.functions import pandas_udf, PandasUDFType
from Data_Pipeline.fill_missing_data import Value_Imputation

class Date_And_Value_Imputation:
    def __init__(self):
        self.value_imputation=Value_Imputation()

    def pyspark_custom_imputation_pipeline(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                       FunctionTransformer(self.value_imputation.cleanup_old))])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data1)

            return transformed_data_df

        transformed_data=df.groupby(analysis_group).apply(transform_features)

        return transformed_data