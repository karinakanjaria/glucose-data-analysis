import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from Data_Pipeline.fill_missing_data import Value_Imputation
from sklearn.preprocessing import FunctionTransformer

from pyspark.sql.functions import pandas_udf, PandasUDFType, lit


class Sklearn_Pipeline:
    def __init__(self):
        self.value_imputation=Value_Imputation()

    ################ PySpark ################
    def pyspark_custom_imputation_pipeline(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId','Value','GlucoseDisplayTime','RecordedSystemTime', 'RecordedDisplayTime', 'GlucoseSystemTime']]

            # Imputation
            custom_imputation=Pipeline(steps=[("custom_imputation",
                                       FunctionTransformer(self.value_imputation.fill_missing_bootstrap))])

            transformed_data1=custom_imputation.fit_transform(df)
            transformed_data_df=pd.DataFrame(transformed_data1)

            return transformed_data_df

        transformed_data=df.groupby(analysis_group).apply(transform_features)

        return transformed_data


    def pyspark_sklearn_pipeline_categorical(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'inserted', 
                    'missing', 'y_Binary', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove']]

            # Categorical Features
            categorical_features=['inserted', 'missing']
            categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor_2=ColumnTransformer([('categorical', categorical_transformer, categorical_features)],
                                            remainder = 'passthrough')

            cat_pipe_pipeline=Pipeline([('preprocessing_2', preprocessor_2)])

            transformed_data1=cat_pipe_pipeline.fit_transform(df)

            transformed_data_df=pd.DataFrame(transformed_data1)

            transformed_data_df['combine_inserted']=transformed_data_df[[0,1]].values.tolist()
            transformed_data_df['combine_missing']=transformed_data_df[[2,3]].values.tolist()
            transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:4],axis = 1)

            transformed_data_df.columns=['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 
                                         'y_Binary', 'Median', 'Mean', 'Std Dev', 
                                        'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing']
            
            return transformed_data_df
        
        transformed_data=df.groupby(analysis_group).apply(transform_features)
        
        return transformed_data



    def pyspark_sklearn_pipeline_numerical(self, df, output_schema, analysis_group):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(pdf):
            df=pdf[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'Median', 'Mean', 'Std Dev', 
                    'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing']]

            # Numerical Features
            numeric_features=['Value', 'Median', 'Mean', 'Std Dev',	'Max', 'Min', 'AreaBelow', 'AreaAbove']
            numeric_transformer=Pipeline([('scaler', StandardScaler())])

            preprocessor_2=ColumnTransformer([('numerical', numeric_transformer, numeric_features)],
                                            remainder = 'passthrough')

            num_pipe_pipeline=Pipeline([('preprocessing_2', preprocessor_2)])

            transformed_data1=num_pipe_pipeline.fit_transform(df)

            transformed_data_df=pd.DataFrame(transformed_data1)
                        
            transformed_data_df.columns=['Value', 'Median', 'Mean', 'Std Dev',	'Max', 'Min', 'AreaBelow', 'AreaAbove', 
                                        'PatientId', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'inserted', 'missing']
            transformed_data_df=transformed_data_df[['Value', 'Median', 'Mean', 'Std Dev',	'Max', 'Min', 'AreaBelow', 'AreaAbove', 
                                        'PatientId', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'inserted', 'missing']]
            
            return transformed_data_df
        
        transformed_data=df.groupby(analysis_group).apply(transform_features)
        
        return transformed_data







    ################ Pandas ################
    def pandas_custom_imputation_pipeline(self, df):
        df=df[['PatientId','Value','GlucoseDisplayTime','RecordedSystemTime', 'RecordedDisplayTime', 'GlucoseSystemTime']]

        # Imputation
        custom_imputation=Pipeline(steps=[("custom_imputation",
                                   FunctionTransformer(self.value_imputation.fill_missing_bootstrap))])


        transformed_data1=custom_imputation.fit_transform(df)

        transformed_data_df=pd.DataFrame(transformed_data1)
                    
        return transformed_data_df


    def pandas_transform_categorical_features(self, df):
        df=df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'inserted', 
                'missing', 'y_Binary', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove']]

        # Categorical Features
        categorical_features=['inserted', 'missing']
        categorical_transformer=Pipeline([('imputer_cat', SimpleImputer(strategy='constant', fill_value=np.nan)),
                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor_2=ColumnTransformer([('categorical', categorical_transformer, categorical_features)],
                                        remainder = 'passthrough')

        cat_pipe_pipeline=Pipeline([('preprocessing_2', preprocessor_2)])

        transformed_data1=cat_pipe_pipeline.fit_transform(df)

        transformed_data_df=pd.DataFrame(transformed_data1)

        transformed_data_df['combine_inserted']=transformed_data_df[[0,1]].values.tolist()
        transformed_data_df['combine_missing']=transformed_data_df[[2,3]].values.tolist()
        transformed_data_df=transformed_data_df.drop(transformed_data_df.iloc[:, 0:4],axis = 1)
  
        transformed_data_df.columns=['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'Median', 'Mean', 'Std Dev', 
                                     'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing']
        transformed_data_df=transformed_data_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'Median', 'Mean', 'Std Dev', 
                                                 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing', 'y_Binary',]]

        return transformed_data_df


    def pandas_transform_numerical_features(self, df):
        df=df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'Median', 'Mean', 'Std Dev', 
                'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing']]

        # Numerical Features
        numeric_features=['Value', 'Median', 'Mean', 'Std Dev',	'Max', 'Min', 'AreaBelow', 'AreaAbove']
        numeric_transformer=Pipeline([('scaler', StandardScaler())])

        preprocessor_2=ColumnTransformer([('numerical', numeric_transformer, numeric_features)],
                                        remainder = 'passthrough')

        num_pipe_pipeline=Pipeline([('preprocessing_2', preprocessor_2)])

        transformed_data1=num_pipe_pipeline.fit_transform(df)

        transformed_data_df=pd.DataFrame(transformed_data1)
                    
        transformed_data_df.columns=['Value', 'Median', 'Mean', 'Std Dev',	'Max', 'Min', 'AreaBelow', 'AreaAbove', 
                                     'PatientId', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'y_Binary', 'inserted', 'missing']
        transformed_data_df=transformed_data_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'Median', 'Mean', 'Std Dev', 
                                                 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'inserted', 'missing', 'y_Binary']]

        return transformed_data_df