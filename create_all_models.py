################################ Libraries ################################
from Input_Variables.read_vars import xgboost_regression_model_storage_location, \
                                      linear_regression_model_storage_location, \
                                      random_forest_regression_model_storage_location, \
                                      factorization_machines_regression_model_storage, \
                                      xgboost_classification_model_storage_location, \
                                      logistic_regression_classification_model_storage_location, \
                                      random_forest_classification_model_storage_location, \
                                      final_xgboost_classification_model_storage_location, \
                                      random_seed
from Read_In_Data.read_data import Reading_Data
from Data_Pipeline.scaling_pipeline import Feature_Transformations
from Model_Creation.regression_models import Create_Regression_Models
from Model_Creation.classification_models import Create_Classification_Models
from Model_Creation.xgbclassifier_model import Create_XGBClassifier_Model
import os


################################ Read In Modules ################################
reading_data=Reading_Data()
feature_transformations=Feature_Transformations()
create_regression_models=Create_Regression_Models()
create_classification_models=Create_Classification_Models()
create_xgbclassifier_model=Create_XGBClassifier_Model()

################################ Regression, Classification, Or Both ################################
train_regression=False
train_classification=False


################################ Read In Data ################################
# Training Summary Stats Data
training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_train_bool_updated'),x),
                                               os.listdir('/cephfs/summary_stats/all_train_bool_updated')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]


# Cross Validation Summary Stats Data
val_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/all_val_bool_updated'), x),
                                          os.listdir('/cephfs/summary_stats/all_val_bool_updated')))
val_files=[i for i in val_files if not ('.crc' in i or 'SUCCESS' in i)]


# Calling DataFrames
summary_stats_train=reading_data.read_in_all_summary_stats(file_list=training_files)
summary_stats_val=reading_data.read_in_all_summary_stats(file_list=val_files)


################################ Combine Train and Cross Validation ################################
df_train_val_combined=summary_stats_train.union(summary_stats_val)
df_train_val_combined.show(2)
print((df_train_val_combined.count(), len(df_train_val_combined.columns)))


################################ Stages: Scaling Using Custom Transformer ################################
pipeline_transformation_stages=feature_transformations.numerical_scaling(df=df_train_val_combined)


################################ Regression Models ################################
if train_regression is True:
    regression_models_storage_locations=[xgboost_regression_model_storage_location, 
                                         linear_regression_model_storage_location,
                                         random_forest_regression_model_storage_location,
                                         factorization_machines_regression_model_storage]
    create_regression_models\
            .regression_modeling(ml_df=df_train_val_combined,
                                 stages=pipeline_transformation_stages, 
                                 random_seed=random_seed,
                                 regression_models_storage_locations=regression_models_storage_locations,
                                 num_folds=3)


################################ Classification Models ################################
if train_classification is True:
    classification_models_storage=[xgboost_classification_model_storage_location,
                                   logistic_regression_classification_model_storage_location,
                                   random_forest_classification_model_storage_location]
    create_classification_models\
            .classification_modeling(ml_df=df_train_val_combined,
                                     stages=pipeline_transformation_stages, 
                                     random_seed=random_seed,
                                     classification_models_storage_locations=classification_models_storage,
                                     num_folds=3)


################################ XGB Classifier Model ################################
else:
    xgbclassifier_model_storage=final_xgboost_classification_model_storage_location
    create_xgbclassifier_model\
            .xgbclassifier_modeling(ml_df=df_train_val_combined,
                                    stages=pipeline_transformation_stages, 
                                    random_seed=random_seed,
                                    xgbclassifier_model_storage_location=xgbclassifier_model_storage,
                                    num_folds=3)
    
# else:
#     print('Did Not Choose To Model Either Regression or Classification Models.')


"""################################ Model Evaluation ################################"""
# model_evaluation=evaluate_model.regression_evaluation(testing_predictions=testing_predictions, 
#                                                       eval_csv_location=evaluation_metrics_output_storage)
# model_evaluation.head()


# ################################ Feature Importance ################################
# feature_importance_df=feature_importance \
#                             .feature_importance_accuracy_gain(xgboost_model=xgboost_regression_model, 
#                                                               feature_importance_storage_location=feature_importance_storage_location)
# feature_importance_df.head(10)


# ################################ Feature Importance Plot ################################
# overall_feature_plot=xgboost_classification_plot.feature_overall_importance_plot(feature_importance_df=feature_importance_df,
#                                                                                  overall_importance_plot_location=overall_feature_importance_plot_location)