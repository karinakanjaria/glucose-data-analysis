from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd

class Evaluate_Model:
    def classification_evaluation(self, testing_predictions, eval_csv_location):
        testing_predictions=testing_predictions \
            .withColumn("y_binary", testing_predictions["y_binary"].cast("double")) \
            .withColumn("prediction", testing_predictions["prediction"].cast("double"))

        metrics_dict={'accuracy': None,
                     'precisionByLabel': None,
                     'recallByLabel': None,
                     'f1': None,
                     'auc': None,
                     'confusion_matrix': None}
        
        for metric in ['accuracy', 'precisionByLabel', 'recallByLabel', 'f1']:
            eval_metric=MulticlassClassificationEvaluator(labelCol="y_binary", 
                                                          predictionCol="prediction", 
                                                          metricName=metric)
            metric_value=eval_metric.evaluate(testing_predictions)
            metrics_dict[metric]=metric_value
    

        eval_auc=BinaryClassificationEvaluator(labelCol="y_binary", 
                                               rawPredictionCol="prediction")
        auc=eval_auc.evaluate(testing_predictions)
        metrics_dict['auc']=auc
        
        preds_and_labels=testing_predictions.select(['prediction','y_binary']).withColumn('label', F.col('y_binary').cast(FloatType())).orderBy('prediction')
        preds_and_labels = preds_and_labels.select(['prediction','label'])
        metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
        cf_matrix=metrics.confusionMatrix().toArray()
        metrics_dict['confusion_matrix']=f'{cf_matrix}'
        
        eval_df=pd.DataFrame(metrics_dict, index=[0])
        eval_df.to_csv(eval_csv_location, index=False, header=True)
        
        return eval_df