from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd

class Model_Predictions_And_Evaluations:
    def regression_create_predictions(self, test_df, model):
        predict_df=model.transform(test_df).select('NumId', 'Chunk', 'prediction', 'DiffPrevious')
        
        return predict_df
    
    def regression_create_evaluations(self, model_type, pipeline_location, test_data, regression_evaluation_metrics_output_storage):
        pipeline_model=PipelineModel.load(pipeline_location)
        testing_predictions=self.regression_create_predictions(test_df=test_data, 
                                                               model=pipeline_model)
        
        evaluators=['rmse', 'mse', 'r2', 'mae', 'var']
        metrics_dict={'rmse': None,
                      'mse': None,
                      'r2': None,
                      'mae': None,
                      'var': None}
        
        for metric in evaluators:
            eval_metric=RegressionEvaluator(labelCol="DiffPrevious", 
                                            predictionCol="prediction", 
                                            metricName=metric)
            metric_value=eval_metric.evaluate(testing_predictions)
            metrics_dict[metric]=metric_value
            
        
        eval_df=pd.DataFrame(metrics_dict, index=[0])
        
        output_location=regression_evaluation_metrics_output_storage+model_type+'eval_metrics.csv'
        eval_df.to_csv(output_location, index=False, header=True)