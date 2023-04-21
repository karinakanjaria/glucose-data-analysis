from xgboost.spark import SparkXGBRegressor, SparkXGBClassifierModel 
# from sparkxgb.xgboost import XGBoostClassificationModel, XGBoostClassifier
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:
    # def xgboost_classifier(self, ml_df, va1, ss, va2, model_storage_location):
    def xgboost_classifier(self, ml_df, stages, model_storage_location):
        # assume the label column is named "class"
        label_name = "y_binary"
        features_col="features"

        # create a xgboost pyspark regressor estimator and set use_gpu=True
        # xgboost_regressor = SparkXGBRegressor(features_col=features_col,
        #                                       label_col=label_name,
        #                                       num_workers=2,
        #                                       use_gpu=True,)
        
        
        xgboost_classifier = SparkXGBClassifierModel(features_col=features_col,
                                               seed=123, 
                                              label_col=label_name,
                                              num_workers=2,
                                              use_gpu=True,)

        
        stages.append(xgboost_classifier)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(model_storage_location)
        
        return model