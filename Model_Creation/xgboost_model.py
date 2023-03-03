from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit

class XGBoost_Classification:
    def pyspark_xgboost(self, df, output_schema, train_split, test_split):
        group_column='PatientId'
        y_column='y_Binary'
        x_columns=['lag_1', 'lag_2', 'lag_3']

        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        # Input/output are both a pandas.DataFrame
        def xgboost_model(pdf):
            group_key=pdf[group_column].iloc[0]

            X=pdf[x_columns]
            y=pdf[y_column]
            
            X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=train_split, test_size=test_split)

            model=XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
            model.fit(X_train, y_train)
            preds=model.predict(X_test)
            # Save as a pickle file
            model.save_model(f'Model_Creation/Saved_Models/PySpark/test_model.json')

            save_results_df=pd.DataFrame()
            save_results_df['Predictions']=preds
            save_results_df['Actual']=y_test.tolist()
            save_results_df['PatientId']=group_key
            save_results_df['GlucoseDisplayTime']=pdf['GlucoseDisplayTime']

            save_results_df=save_results_df[['PatientId', 'GlucoseDisplayTime', 'Predictions', 'Actual']]

            return save_results_df

        preds_df=df.groupby(group_column).apply(xgboost_model)
        preds_df.write.format("csv").mode('overwrite').save("Model_Creation/Saved_Results/PySpark")

        return preds_df


    def pandas_xgboost(self, df, output_schema, train_split, test_split):
        return None