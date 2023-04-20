from xgboost.spark import SparkXGBRegressor

class Create_PySpark_XGBoost:
    def xgboost_classifier(self, ml_df, model_storage_location):
        # assume the label column is named "class"
        label_name = "y_binary"
        features_col="features"

        # create a xgboost pyspark regressor estimator and set use_gpu=True
        regressor = SparkXGBRegressor(features_col=features_col,
                                      label_col=label_name,
                                      num_workers=2,
                                      use_gpu=True,)

        # train and return the model
        model = regressor.fit(ml_df)
        
        model.save(model_storage_location)

        # predict on test data
        predict_df = model.transform(ml_df)
        
        return predict_df