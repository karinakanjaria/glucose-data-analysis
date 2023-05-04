from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd

class Evaluate_Model:
    def regression_evaluation(self, testing_predictions, eval_csv_location):
        evaluators=['rmse', 'mse', 'r2', 'mae', 'var']

        metrics_dict={'rmse': None,
                     'mse': None,
                     'r2': None,
                     'mae': None,
                     'var': None}
        
        for metric in evaluators:
            eval_metric=RegressionEvaluator(labelCol="target", 
                                            predictionCol="prediction", 
                                            metricName=metric)
            metric_value=eval_metric.evaluate(testing_predictions)
            metrics_dict[metric]=metric_value
            
        
        eval_df=pd.DataFrame(metrics_dict, index=[0])
        eval_df.to_csv(eval_csv_location, index=False, header=True)
        
        return eval_df