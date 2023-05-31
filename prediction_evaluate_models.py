################################ Libraries ################################
from Input_Variables.read_vars import xgboost_regression_model_storage_location, \
                                      linear_regression_model_storage_location, \
                                      random_forest_regression_model_storage_location, \
                                      factorization_machines_regression_model_storage, \
                                      xgboost_classification_model_storage_location, \
                                      logistic_regression_classification_model_storage_location, \
                                      random_forest_classification_model_storage_location, \
                                      regression_evaluation_metrics_output_storage, \
                                      classification_evaluation_metrics_output_storage
from Read_In_Data.read_data import Reading_Data
from Model_Preds_Eval.pyspark_model_preds_and_eval import Model_Predictions_And_Evaluations

# from Feature_Importance.model_feature_importance import Feature_Importance
# from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot
import os


################################ Read In Modules ################################
reading_data=Reading_Data()
model_predictions_and_evaluations=Model_Predictions_And_Evaluations()
# model_predictions=Model_Predictions()
# evaluate_model=Evaluate_Model()
# feature_importance=Feature_Importance()
# xgboost_classification_plot=XGBoost_Classification_Plot()


################################ Read In Data ################################
# Testing Summary Stats Data
test_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_test_bool_updated'), x),
                                           os.listdir('/cephfs/summary_stats/all_test_bool_updated')))
test_files=[i for i in test_files if not ('.crc' in i or 'SUCCESS' in i)]

# Calling DataFrames
summary_stats_test=reading_data.read_in_all_summary_stats(file_list=test_files)
summary_stats_test.show(2)
print((summary_stats_test.count(), len(summary_stats_test.columns)))


################################ Regression, Classification, Or Both ################################
train_regression=False
train_classification=True


################################ Regression: Testing Predictions and Evaluations ################################
if train_regression is True:
    regression_models_pipeline_locations={'XGBoost': xgboost_regression_model_storage_location, 
                                          'Linear_Regression': linear_regression_model_storage_location,
                                          'Random_Forest': random_forest_regression_model_storage_location,
                                          'Factorization_Machines': factorization_machines_regression_model_storage}

    for regression_type in regression_models_pipeline_locations:
        print(f'Regression: Completing {regression_type} Model Evaluations')
        model_predictions_and_evaluations.regression_create_evaluations(model_type=regression_type, 
                                                                        pipeline_location=regression_models_pipeline_locations[regression_type],
                                                                        test_data=summary_stats_test,
                                                                        regression_evaluation_metrics_output_storage=regression_evaluation_metrics_output_storage)

        
################################ Classification: Testing Predictions and Evaluations ################################
elif train_classification is True:
    classification_models_pipeline_locations={'XGBoost': xgboost_classification_model_storage_location, 
                                              'Logistic_Regression': logistic_regression_classification_model_storage_location,
                                              'Random_Forest': random_forest_classification_model_storage_location}
    
    for classification_type in classification_models_pipeline_locations:
        print(f'Classification: Completing {classification_type} Model Evaluations')
        
        model_predictions_and_evaluations.classification_create_evaluations(model_type=classification_type, 
                                                                            pipeline_location=classification_models_pipeline_locations[classification_type], 
                                                                            test_data=summary_stats_test, 
                                                                            classification_evaluation_metrics_output_storage=classification_evaluation_metrics_output_storage)


else:
    print('Did Not Choose To Evaluate Either Regression or Classification Models.')



# ################################ Feature Importance ################################
# feature_importance_df=feature_importance \
#                             .feature_importance_accuracy_gain(xgboost_model=xgboost_regression_model, 
#                                                               feature_importance_storage_location=feature_importance_storage_location)
# feature_importance_df.head(10)


# ################################ Feature Importance Plot ################################
# overall_feature_plot=xgboost_classification_plot.feature_overall_importance_plot(feature_importance_df=feature_importance_df,
#                                                                                  overall_importance_plot_location=overall_feature_importance_plot_location)