from xgboost.spark import SparkXGBClassifier

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

class Create_Classification_Models:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        self.prediction_column_name="prediction"
            
    def classification_modeling(self, ml_df, stages, random_seed, classification_models_storage_locations, num_folds):
        location_counter=0
        model_types=[]
        model_mapping={'XGBoost': }
        
        ml_df=ml_df.withColumn("foldCol", ml_df.NumId % num_folds)
        
        evaluator_rmse=RegressionEvaluator(metricName='rmse',
                                           labelCol=self.label_name,
                                           predictionCol=self.prediction_column_name)
        paramGrid=ParamGridBuilder().build()
        
        for model_type in model_types:
            if location_counter > 0:
                stages.pop()
                print(f'Currently on {model_type} Model')
            else:
                print(f'Currently on {model_type} Model')
            crossval=CrossValidator(estimator=model_mapping[model_type],
                                    evaluator=evaluator_rmse,
                                    estimatorParamMaps=paramGrid,
                                    foldCol='foldCol',
                                    collectSubModels=False)

            print('Cross Validation Occuring')
            stages.append(crossval)
            pipeline=Pipeline(stages=stages)

            model=pipeline.fit(ml_df)

            model.write().overwrite().save(regression_models_storage_locations[location_counter])
            print(f'Model Saved to {regression_models_storage_locations[location_counter]}')
            location_counter+=1
        
        return None