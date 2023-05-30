from pyspark.sql import functions as f

class Model_Predictions:
    def create_predictions_with_model(self, test_df, model):
        predict_df=model.transform(test_df).select('NumId', 'Chunk', 'DiffPrevious', 'prediction')
        
        return predict_df    