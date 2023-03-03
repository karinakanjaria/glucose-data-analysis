from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyspark.sql.functions import pandas_udf, PandasUDFType, lit

import pandas as pd

class Classification_Evalaution_Metrics:
    def pyspark_classification_model_evaluation_metrics(self, df, output_schema):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        # Input/output are both a pandas.DataFrame
        def classification_accuracy(pdf):
            group_key=pdf['PatientId'].iloc[0]

            metric_dict={'Classification_Accuracy_Score': [accuracy_score(pdf['Predictions'], pdf['Actual'])],
                         'Precision_Score': [precision_score(pdf['Predictions'], pdf['Actual'])],
                         'Recall_Score': [recall_score(pdf['Predictions'], pdf['Actual'])],
                         'F1_Score': [f1_score(pdf['Predictions'], pdf['Actual'])],
                         'Confusion_Matrix': [confusion_matrix(pdf['Predictions'], pdf['Actual']).tolist()]}


            class_acc_value=pd.DataFrame(metric_dict)
            class_acc_value['PatientId']=group_key

            class_acc_value=class_acc_value[['PatientId', 'Classification_Accuracy_Score', 'Precision_Score', 'Recall_Score', 'F1_Score', 'Confusion_Matrix']]

            return class_acc_value

        accuracy_df=df.groupby('PatientId').apply(classification_accuracy)

        return accuracy_df