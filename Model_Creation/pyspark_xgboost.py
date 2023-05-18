# from xgboost.spark import SparkXGBClassifier
from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        
    
    def initial_training_xgboost_regression(self, ml_df, stages, random_seed):
        # xgb_regression=SparkXGBRegressor(features_col=features_col, 
        #                                   label_col=label_name,
        #                                   num_workers=4,
        #                                   random_state=random_seed,
        #                                   use_gpu=True)
        
        initial_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
                                                 label_col=self.label_name,
                                                 random_state=random_seed,
                                                 use_gpu=False)

        stages.append(initial_xgb_regression)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        # model.write().overwrite().save(model_storage_location)
        
        return model
    
    
    def batch_training_xgboost_regression(self, ml_df, stages, random_seed, past_model):
        batch_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
                                               label_col=self.label_name,
                                               random_state=random_seed,
                                               use_gpu=False,
                                               xgb_model=past_model.stages[-1].get_booster())

        stages.append(batch_xgb_regression)
        pipeline=Pipeline(stages=stages)
        new_model=pipeline.fit(ml_df)
       
        # new_model.write().overwrite().save(model_storage_location)
        
        return new_model