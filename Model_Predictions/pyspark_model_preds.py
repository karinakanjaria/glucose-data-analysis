from pyspark.sql import functions as f

class Model_Predictions:
    def create_predictions_with_model(self, test_df, model):
        predict_df=model.transform(test_df).select('PatientId', 
                                                   'GlucoseDisplayTime',
                                                   'Value',
                                                   "y_binary", 
                                                   'prediction')
        
        predict_df=predict_df.withColumn('prediction_label', 
                                         f.when(f.col('prediction') > 0.50, 1).otherwise(0))
        
        return predict_df    