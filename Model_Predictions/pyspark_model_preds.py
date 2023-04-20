class Model_Predictions:
    def create_predictions_with_model(self, df, model):
        predict_df = model.transform(df)
        
        return predict_df    