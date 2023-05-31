from xgboost.spark import SparkXGBClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import FMClassifier

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


class Create_Classification_Models:
    def __init__(self):
        self.features_col="features"
        self.label_name="target"
        self.prediction_column_name="prediction"
            
    def classification_modeling(self, ml_df, stages, random_seed, classification_models_storage_locations, num_folds):
        mapping= {0: 0, 1: 1, -1: 2}
        ml_df=ml_df.replace(to_replace=mapping, subset=['target'])
        
        location_counter=0
        model_types=['XGBoost', 'Logistic_Regression', 'Random_Forest']
        model_mapping={'XGBoost': SparkXGBClassifier(features_col=self.features_col, 
                                                     label_col=self.label_name,
                                                     random_state=random_seed,
                                                     use_gpu=True),
                       
                       'Logistic_Regression': LogisticRegression(featuresCol=self.features_col, 
                                                                 labelCol=self.label_name,
                                                                 standardization=False),
                       
                       'Random_Forest': RandomForestClassifier(featuresCol=self.features_col, 
                                                               labelCol=self.label_name,
                                                               seed=random_seed)
                      }
        
        ml_df=ml_df.withColumn("foldCol", ml_df.NumId % num_folds)
        
        evaluator_logloss=MulticlassClassificationEvaluator(metricName='logLoss',
                                                            labelCol=self.label_name,
                                                            predictionCol=self.prediction_column_name)
        paramGrid=ParamGridBuilder().build()
        
        for model_type in model_types:
            if location_counter > 0:
                stages.pop()
                print(f'Currently on {model_type} Model')
            else:
                print(f'Currently on {model_type} Model')
            crossval=CrossValidator(estimator=model_mapping[model_type],
                                    evaluator=evaluator_logloss,
                                    estimatorParamMaps=paramGrid,
                                    foldCol='foldCol',
                                    collectSubModels=False)

            print('Cross Validation Occuring')
            stages.append(crossval)
            pipeline=Pipeline(stages=stages)

            model=pipeline.fit(ml_df)

            model.write().overwrite().save(classification_models_storage_locations[location_counter])
            print(f'Model Saved to {classification_models_storage_locations[location_counter]}')
            location_counter+=1
        
        return None
    
# Evaluation Metrics  
# (f1
#  accuracy
#  weightedPrecision
#  weightedRecall
#  weightedTruePositiveRate
#  weightedFalsePositiveRate
#  weightedFMeasure
#  truePositiveRateByLabel
#  falsePositiveRateByLabel
#  precisionByLabel
#  recallByLabel
#  fMeasureByLabel
#  logLoss
#  hammingLoss)