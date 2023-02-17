import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from pyspark.sql.functions import pandas_udf, PandasUDFType, lit

class Sklearn_Pipeline:
    def pyspark_sklearn_pipeline(self, df, output_schema):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId','Value','GlucoseDisplayTimeRaw','TrendArrow','TrendRate']]
            
            # Categorical Features
            categorical_features=['TrendArrow']
            categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Numerical Features --> Imputation Method: median; Scaling Method: standardization method
            numeric_features=['Value', 'TrendRate']
            numeric_transformer=Pipeline([('imputer_num', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())])

            preprocessor=ColumnTransformer([('categorical', categorical_transformer, categorical_features),
                                            ('numerical', numeric_transformer, numeric_features)],
                                            remainder = 'passthrough')

            pipeline=Pipeline([('preprocessing', preprocessor)])

            transformed_data_array=pipeline.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data_array)

            transformed_data_df['combine']=transformed_data_df[[0,1,2,3,4,5,6]].values.tolist()
            transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:7],axis = 1)
            transformed_data_df.columns=['Value', 'TrendRate', 'PatientId', 'GlucoseDisplayTimeRaw', 'TrendArrow']
            
            return transformed_data_df
        
        df=df.withColumn('Group', lit(1))
        transformed_data=df.groupby('Group').apply(transform_features)
        
        return transformed_data



    def pandas_transform_features(self, df):
        df=df[['PatientId','Value','GlucoseDisplayTimeRaw','TrendArrow','TrendRate']]

        # Categorical Features
        categorical_features=['TrendArrow']
        categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Numerical Features --> Imputation Method: median; Scaling Method: standardization method
        numeric_features=['Value', 'TrendRate']
        numeric_transformer=Pipeline([('imputer_num', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())])

        preprocessor=ColumnTransformer([('categorical', categorical_transformer, categorical_features),
                                        ('numerical', numeric_transformer, numeric_features)],
                                        remainder = 'passthrough')

        pipeline=Pipeline([('preprocessing', preprocessor)])

        transformed_data_array=pipeline.fit_transform(df)
        transformed_data_df=pd.DataFrame(transformed_data_array)

        transformed_data_df['combine']=transformed_data_df[[0,1,2,3,4,5,6]].values.tolist()
        transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:7],axis = 1)
        
        

    def removeDuplicatesByPatientAndTime(data):
    	return data[data.duplicated(subset=['PatientId', 'GlucoseDisplayTime']) == False]

        transformed_data_df.columns=['Value', 'TrendRate', 'PatientId', 'GlucoseDisplayTimeRaw', 'TrendArrow']

        return transformed_data_df
