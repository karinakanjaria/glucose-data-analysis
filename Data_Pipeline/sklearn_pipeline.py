import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from Data_Pipeline.fill_missing_trial import Value_Imputation
from sklearn.preprocessing import FunctionTransformer

from pyspark.sql.functions import pandas_udf, PandasUDFType, lit

class Sklearn_Pipeline:
    def __init__(self):
        self.value_imputation=Value_Imputation()

    def pyspark_sklearn_pipeline(self, df, output_schema):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId','Value','GlucoseDisplayTime','RecordedSystemTime', 'RecordedDisplayTime', 'GlucoseSystemTime','TrendArrow']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                    FunctionTransformer(self.value_imputation.fill_missing_bootstrap))])


            # Categorical Features
            categorical_features=['inserted', 'missing']
            categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Numerical Features
            numeric_features=['Value']
            numeric_transformer=Pipeline([('scaler', StandardScaler())])


            preprocessor_2=ColumnTransformer([('categorical', categorical_transformer, categorical_features),
                                            ('numerical', numeric_transformer, numeric_features)],
                                            remainder = 'passthrough')

            pipeline2=Pipeline([('preprocessing_2', preprocessor_2)])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data2=pipeline2.fit_transform(transformed_data1)

            transformed_data_df=pd.DataFrame(transformed_data2)

            transformed_data_df['combine_inserted']=transformed_data_df[[0,1]].values.tolist()
            transformed_data_df['combine_missing']=transformed_data_df[[2,3]].values.tolist()
            transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:4],axis = 1)
                        
            transformed_data_df.columns=['Value', 'PatientId', 'GlucoseDisplayTime', 'inserted', 'missing']
            transformed_data_df=transformed_data_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'inserted', 'missing']]
            
            return transformed_data_df
        
        df=df.withColumn('Group', lit(1))
        transformed_data=df.groupby('Group').apply(transform_features)
        
        return transformed_data

    def pandas_transform_features(self, df):
        df=df[['PatientId','Value','GlucoseDisplayTime','RecordedSystemTime', 'RecordedDisplayTime', 'GlucoseSystemTime','TrendArrow']]

        # Imputation
        custom_imputation=Pipeline(steps=[("custom_imputation",
                                   FunctionTransformer(self.value_imputation.fill_missing_bootstrap))])


        # Categorical Features
        categorical_features=['inserted', 'missing']
        categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Numerical Features
        numeric_features=['Value']
        numeric_transformer=Pipeline([('scaler', StandardScaler())])


        preprocessor_2=ColumnTransformer([('categorical', categorical_transformer, categorical_features),
                                          ('numerical', numeric_transformer, numeric_features)],
                                         remainder = 'passthrough')

        pipeline2=Pipeline([('preprocessing_2', preprocessor_2)])

        transformed_data1=custom_imputation.fit_transform(df)
        transformed_data2=pipeline2.fit_transform(transformed_data1)

        transformed_data_df=pd.DataFrame(transformed_data2)

        transformed_data_df['combine_inserted']=transformed_data_df[[0,1]].values.tolist()
        transformed_data_df['combine_missing']=transformed_data_df[[2,3]].values.tolist()
        transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:4],axis = 1)
                    
        transformed_data_df.columns=['Value', 'PatientId', 'GlucoseDisplayTime', 'inserted', 'missing']
        transformed_data_df=transformed_data_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'inserted', 'missing']]

        return transformed_data_df