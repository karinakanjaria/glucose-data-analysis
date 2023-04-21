from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline

class Create_PySpark_XGBoost:
    def xgboost_classifier(self, ml_df, stages, model_storage_location):
        features_col="features"
        label_name="y_binary"
        
        xgb_classifier=SparkXGBClassifier(features_col=features_col, 
                                          label_col=label_name,
                                          num_workers=2,
                                          use_gpu=True)

        stages.append(xgb_classifier)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        model.write().overwrite().save(model_storage_location)
        
        return model