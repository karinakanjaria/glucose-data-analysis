################################ Libraries ################################
from Input_Variables.read_vars import xgb_reg_model_storage_location, xgb_class_model_storage_location, random_seed, \
                                      evaluation_metrics_output_storage, \
                                      feature_importance_storage_location, \
                                      overall_feature_importance_plot_location
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.scaling_pipeline import Feature_Transformations
from Model_Creation.pyspark_xgboost import Create_PySpark_XGBoost
from Model_Predictions.pyspark_model_preds import Model_Predictions
from Model_Evaluation.pyspark_model_eval import Evaluate_Model
from Feature_Importance.model_feature_importance import Feature_Importance
from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot
import os


################################ Read In Modules ################################
reading_data=Reading_Data()
feature_transformations=Feature_Transformations()
create_pyspark_xgboost=Create_PySpark_XGBoost()
model_predictions=Model_Predictions()
evaluate_model=Evaluate_Model()
feature_importance=Feature_Importance()
xgboost_classification_plot=XGBoost_Classification_Plot()


################################ Read In Data ################################
# Training Summary Stats Data
training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_train_bool'), x),os.listdir('/cephfs/summary_stats/all_train_bool')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]


# Cross Validation Summary Stats Data
val_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_val_bool'), x),os.listdir('/cephfs/summary_stats/all_val_bool')))
val_files=[i for i in val_files if not ('.crc' in i or 'SUCCESS' in i)]


# Testing Summary Stats Data
test_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_test_bool'), x),os.listdir('/cephfs/summary_stats/all_test_bool')))
test_files=[i for i in test_files if not ('.crc' in i or 'SUCCESS' in i)]

# Calling DataFrames
summary_stats_train=reading_data.read_in_all_summary_stats(file_list=training_files)
summary_stats_val=reading_data.read_in_all_summary_stats(file_list=val_files)
summary_stats_test=reading_data.read_in_all_summary_stats(file_list=test_files)


################################ Combine Train and Cross Validation ################################
df_train_val_combined=summary_stats_train.union(summary_stats_val)
df_train_val_combined.show(2)
print((df_train_val_combined.count(), len(df_train_val_combined.columns)))


################################ Stages: Scaling Using Custom Transformer ################################
pipeline_transformation_stages=feature_transformations.numerical_scaling(df=df_train_val_combined)


################################ XGBoost Regression Model ################################
k_folds=3
xgboost_regression_model=create_pyspark_xgboost\
        .initial_training_xgboost_regression(ml_df=summary_stats_train,
                                             stages=pipeline_transformation_stages, 
                                             random_seed=random_seed,
                                             xgb_reg_model_storage_location=xgb_reg_model_storage_location,
                                             k=k_folds)


################################ Testing Predictions ################################
testing_predictions=model_predictions.create_predictions_with_model(test_df=summary_stats_test, 
                                                                    model=xgboost_regression_model)
testing_predictions.show(10)


################################ Model Evaluation ################################
model_evaluation=evaluate_model.regression_evaluation(testing_predictions=testing_predictions, 
                                                      eval_csv_location=evaluation_metrics_output_storage)
model_evaluation.head()


################################ Feature Importance ################################
feature_importance_df=feature_importance \
                            .feature_importance_accuracy_gain(xgboost_model=xgboost_regression_model, 
                                                              feature_importance_storage_location=feature_importance_storage_location)
feature_importance_df.head(10)


################################ Feature Importance Plot ################################
overall_feature_plot=xgboost_classification_plot.feature_overall_importance_plot(feature_importance_df=feature_importance_df,
                                                                                 overall_importance_plot_location=overall_feature_importance_plot_location)