import os
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class XGBoost_Classification_Plot:
    def read_model_plot_variance(self, model_storage_location):
        model_list=os.listdir(f'{model_storage_location}')

        for individual_model in model_list:
            model=XGBClassifier()
            model.load_model(f'{model_storage_location}/{individual_model}')
            feature_importance=model.get_booster().get_score(importance_type='gain')

            feature_importance_df=pd.DataFrame([feature_importance])

            plot_title=individual_model.replace(".json", "")
            names=list(feature_importance_df.columns)
            values=feature_importance_df.values.tolist()[0]

            plt.bar(np.arange(len(values)), values, tick_label=names)
            plt.xlabel('Feature') 
            plt.ylabel('Information Gain') 
            plt.title(plot_title)
            plt.savefig(f'Model_Plots/Information_Gain_Plots/info_gain_{individual_model}.png')
            plt.show()
            plt.clf()