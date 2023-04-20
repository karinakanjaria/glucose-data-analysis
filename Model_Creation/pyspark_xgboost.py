from xgboost.spark import SparkXGBRegressor

class Create_PySpark_XGBoost:
    def xgboost_classifier(self, ml_df):
        # assume the label column is named "class"
        label_name = "y_binary"

        # get a list with feature column names
        all_col_features=ml_df.columns
        scaled_features=[x for x in all_col_features if "scaled" in x]

        # create a xgboost pyspark regressor estimator and set use_gpu=True
        regressor = SparkXGBRegressor(features_col=scaled_features,
                                      label_col=label_name,
                                      num_workers=2,
                                      use_gpu=True,)

        # train and return the model
        model = regressor.fit(ml_df)
        
        model.save('test1')

        # predict on test data
        predict_df = model.transform(ml_df)
        
        return predict_df