import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType, StructField, FloatType, IntegerType

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, FloatType, LongType
from pyspark.sql.functions import col, mean, stddev
from pyspark.sql import Window

import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable 

from xgboost.spark import SparkXGBRegressor
from pyspark.ml import Pipeline


from Input_Variables.read_vars import evaluation_metrics_output_storage, \
                                      feature_importance_storage_location, \
                                      overall_feature_importance_plot_location


from Model_Predictions.pyspark_model_preds import Model_Predictions
from Model_Evaluation.pyspark_model_eval import Evaluate_Model
from Feature_Importance.model_feature_importance import Feature_Importance
from Model_Plots.xgboost_classification_plots import XGBoost_Classification_Plot

model_predictions=Model_Predictions()
evaluate_model=Evaluate_Model()
feature_importance=Feature_Importance()
xgboost_classification_plot=XGBoost_Classification_Plot()

training_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/train'), x),os.listdir('/cephfs/summary_stats/train')))
training_files=[i for i in training_files if not ('.crc' in i or 'SUCCESS' in i)]

validation_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/val'), x),os.listdir('/cephfs/summary_stats/val')))
validation_files=[i for i in validation_files if not ('.crc' in i or 'SUCCESS' in i)]

test_files=list(map(lambda x: os.path.join(os.path.abspath('/cephfs/summary_stats/test'), x),os.listdir('/cephfs/summary_stats/test')))
test_files=[i for i in test_files if not ('.crc' in i or 'SUCCESS' in i)]



class Reading_Data:
    def __init__(self):
        self.spark = SparkSession.builder.appName("Glucose").getOrCreate()
    
    def read_in_all_summary_stats(self, file_list):        
        summary_stats_df = self.spark.read \
                               .parquet(*file_list)



        return summary_stats_df

read_data=Reading_Data()
summary_stats_train=read_data.read_in_all_summary_stats(file_list=training_files)
summary_stats_train.show(2)
print((summary_stats_train.count(), len(summary_stats_train.columns)))

summary_stats_val=read_data.read_in_all_summary_stats(file_list=validation_files)
summary_stats_val.show(2)
print((summary_stats_val.count(), len(summary_stats_val.columns)))


# summary_stats_test=read_data.read_in_all_summary_stats(file_list=test_files)
# summary_stats_test.show(2)
# print((summary_stats_test.count(), len(summary_stats_test.columns)))


# from pyspark.sql.functions import countDistinct
# df2=summary_stats_train.select(countDistinct("NumId"))
# df2.show()

# df2=summary_stats_val.select(countDistinct("NumId"))
# df2.show()

# df2=summary_stats_test.select(countDistinct("NumId"))
# df2.show()


    
class ColumnScaler(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def _transform(self, df):
        double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
        float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
        long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

        all_numerical=list(set(double_cols+float_cols+long_cols))
        all_numerical.remove('target')
        
        for num_column in all_numerical:
            input_col = f"{num_column}"
            output_col = f"scaled_{num_column}"

            w = Window.partitionBy('NumId')

            mu = mean(input_col).over(w)
            sigma = stddev(input_col).over(w)

            df=df.withColumn(output_col, (col(input_col) - mu)/(sigma))
            
        return df


class Feature_Transformations:    
    def numerical_scaling(self, df):
#         double_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType)]
#         float_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, FloatType)]
#         long_cols=[f.name for f in df.schema.fields if isinstance(f.dataType, LongType)]

#         all_numerical=list(set(double_cols+float_cols+long_cols))
#         all_numerical.remove('target')
        all_numerical=['Mean', 'StdDev', 'Median', 'Min', 'Max', 'AvgFirstDiff', 'AvgSecDiff', 'StdFirstDiff', 'StdSecDiff', 'CountAbove', 'CountBelow', 'TotalOutOfRange']

        featureArr = [('scaled_' + f) for f in all_numerical]

        columns_scaler=ColumnScaler()
    
        va2 = VectorAssembler(inputCols=featureArr, outputCol="features", handleInvalid='skip')

        stages=[columns_scaler]+[va2]
        
        return stages
    
feature_transformations=Feature_Transformations()
pipeline_transformation_stages=feature_transformations.numerical_scaling(df=summary_stats_train)





class Create_PySpark_XGBoost:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        
    
    def initial_training_xgboost_regression(self, ml_df, stages, random_seed):
        # xgb_regression=SparkXGBRegressor(features_col=features_col, 
        #                                   label_col=label_name,
        #                                   num_workers=4,
        #                                   random_state=random_seed,
        #                                   use_gpu=True)
        
        initial_xgb_regression=SparkXGBRegressor(features_col=self.features_col, 
                                                 label_col=self.label_name,
                                                 random_state=random_seed,
                                                 use_gpu=False)

        stages.append(initial_xgb_regression)
        pipeline=Pipeline(stages=stages)
        
        model=pipeline.fit(ml_df)
        
        return model
    
create_pyspark_xgboost=Create_PySpark_XGBoost()
xgboost_regression_model=create_pyspark_xgboost.initial_training_xgboost_regression(ml_df=summary_stats_train, 
                                                                                    stages=pipeline_transformation_stages, 
                                                                                    random_seed=123)

model_storage_location='/cephfs/Saved_Models/Summary_Stats_Model'
xgboost_regression_model.write().overwrite().save(model_storage_location)






testing_predictions=model_predictions.create_predictions_with_model(test_df=summary_stats_val, 
                                                                    model=xgboost_regression_model)
testing_predictions.show(10)



model_evaluation=evaluate_model.regression_evaluation(testing_predictions=testing_predictions, 
                                                      eval_csv_location=evaluation_metrics_output_storage)
model_evaluation.head()


feature_importance_df=feature_importance \
                            .feature_importance_accuracy_gain(xgboost_model=xgboost_regression_model, 
                                                              feature_importance_storage_location=feature_importance_storage_location)
feature_importance_df.head(10)


overall_feature_plot=xgboost_classification_plot.feature_overall_importance_plot(feature_importance_df=feature_importance_df,
                                                                                 overall_importance_plot_location=overall_feature_importance_plot_location)