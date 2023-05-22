from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        
    def initial_training_xgboost_regression(self, ml_df, stages, random_seed, model_storage_location):        
        initial_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
                                                 label_col=self.label_name,
                                                 random_state=random_seed,
                                                 use_gpu=False)

        stages.append(initial_xgb_regression)
        pipeline=Pipeline(stages=stages)
        print('Model Created')
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(model_storage_location)
        print(f'Model Saved to {model_storage_location}')
        
        return model