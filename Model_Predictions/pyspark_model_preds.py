from pyspark.sql import functions as f

class Model_Predictions:
    def create_predictions_with_model(self, test_df, model):
        predict_df=model.transform(test_df).select('PatientId', 
                                                   'GlucoseDisplayTime',
                                                   'Value',
                                                   'y_binary',
                                                   'prediction',
                                                   'rawPrediction',
                                                   'probability')
        
        return predict_df    