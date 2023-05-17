# from xgboost.spark import SparkXGBClassifier
from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:
    def initial_training_xgboost_regression(self, ml_df, stages, model_storage_location, random_seed):
        features_col="features"
        label_name="target"

        xgb_regression=SparkXGBRegressor(features_col=features_col, 
                                          label_col=label_name,
                                          num_workers=4,
                                          random_state=random_seed,
                                          use_gpu=True)

        
        
        stages.append(xgb_regression)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(model_storage_location)
        
        return model
    
    
    def batch_training_xgboost_regression(self, ml_df, stages, model_storage_location, random_seed):
        features_col="features"
        label_name="target"

        xgb_classifier=SparkXGBRegressor(features_col=features_col, 
                                          label_col=label_name,
                                          num_workers=4,
                                          random_state=random_seed,
                                          use_gpu=True)

        
        
        stages.append(xgb_classifier)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(model_storage_location)
        
        return model