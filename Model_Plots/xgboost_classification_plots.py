import plotly.express as px
import pandas as pd

class XGBoost_Classification_Plot:
    def feature_importance_plot(self, feature_importance_df):
        

        fig = px.bar(feature_importance_df, 
                     x='Accuracy Gain', 
                     y='Feature', 
                     orientation='h', 
                     color='Accuracy Gain')
        fig.write_image("fig1.png")