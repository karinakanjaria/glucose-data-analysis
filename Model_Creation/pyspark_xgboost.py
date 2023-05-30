from xgboost.spark import SparkXGBRegressor
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:        
    def initial_training_xgboost_regression(self, ml_df, stages, random_seed, xgb_reg_model_storage_location):  
        features_col="features"
        label_name="DiffPrevious"
        
        xgb_regression=SparkXGBRegressor(features_col=features_col, 
                                         label_col=label_name,
                                         random_state=random_seed,
                                         use_gpu=True)
        
        # paramGrid=ParamGridBuilder().addGrid(xgb_regression.max_depth,[3, 6, 10])\
        # .addGrid(xgb_regression.n_estimators,[10, 50, 100])\
        # .addGrid(xgb_regression.subsample,[0.6, 0.8, 1.0])\
        # .build()
        paramGrid=ParamGridBuilder().addGrid(xgb_regression.max_depth,[6, 10])\
        .addGrid(xgb_regression.n_estimators,[2, 3])\
        .build()
        
        
        
        evaluator_rmse=RegressionEvaluator(metricName='rmse',
                                           labelCol="DiffPrevious",
                                           predictionCol="prediction")
        
        crossval=CrossValidator(estimator=xgb_regression,
                                estimatorParamMaps=paramGrid,
                                evaluator=evaluator_rmse,
                                numFolds=2,
                                collectSubModels=True)
        
        print('Cross Validation and Hyperparameter Tuning Occuring')
        stages.append(crossval)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(xgb_reg_model_storage_location)
        print(f'Model Saved to {xgb_reg_model_storage_location}')
        
        return model
    
    
    
#     def initial_training_xgboost_classification(self, ml_df, stages, random_seed, xgb_class_model_storage_location):
#         features_col="features"
#         label_name="target"
        
#         initial_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
#                                                  label_col=self.label_name,
#                                                  random_state=random_seed,
#                                                  use_gpu=False)

#         stages.append(initial_xgb_regression)
#         pipeline=Pipeline(stages=stages)
#         print('Model Created')
        
#         model=pipeline.fit(ml_df)
        
#         model.write().overwrite().save(model_storage_location)
#         print(f'Model Saved to {model_storage_location}')
        
#         return model