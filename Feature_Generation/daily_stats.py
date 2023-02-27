# Python Libraries
import numpy as np
import pandas as pd

from pyspark.sql.functions import pandas_udf, PandasUDFType, lit

class Daily_Stats_Features:
    def pyspark_sklearn_pipeline(self, df, output_schema, lower, upper):
        @pandas_udf(output_schema, PandasUDFType.GROUPED_MAP)
        def transform_features(data):
            median_df = pd.DataFrame(columns=['PatientId', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'Date'])
            dates = data.GlucoseDisplayDate.unique()
            patients=data.PatientId.unique()
            
            for patient in patients:
                for date in dates:
                    date_sub = data.loc[(data.GlucoseDisplayDate == date) & (data.PatientId == patient)]
                    
                    # obtain summary values
                    median = np.nanmedian(date_sub.Value)
                    mean = np.nanmean(date_sub.Value)
                    std = np.nanstd(date_sub.Value)
                    maxVal = np.nanmax(date_sub.Value)
                    minVal = np.nanmin(date_sub.Value)
                    
                    # obtain areas above and below recommendation
                    upperSegs = date_sub.loc[date_sub.Value > upper].Value - upper
                    areaAbove = np.nansum(upperSegs)
                    lowerSegs = -(date_sub.loc[date_sub.Value < lower].Value - lower)
                    areaBelow = np.nansum(lowerSegs)
                    
                    sample = [date_sub.iloc[0]["PatientId"], median, mean, std, maxVal, minVal, areaBelow, areaAbove, date]
                    
                    median_df.loc[len(median_df.index)] = sample
            
            merged_median_df=data.merge(median_df, left_on=['PatientId', 'GlucoseDisplayDate'], right_on=['PatientId', 'Date'], how='left')
            merged_median_df=merged_median_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate',
                                               'inserted', 'missing', 'Median', 'Mean', 'Std Dev', 'Max', 'Min',
                                               'AreaBelow', 'AreaAbove']]
            return merged_median_df
        
        df=df.withColumn('Group', lit(1))
        added_daily_features=df.groupby('Group').apply(transform_features)
        
        return added_daily_features

    def pandas_compressDailyValues(self, data, lower, upper):
        '''Keyword arguments:
        data -- DataFrame of a single patient with no missing data'''
                
        median_df = pd.DataFrame(columns=['PatientId', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'Date'])
        dates = data.GlucoseDisplayDate.unique()
        patients=data.PatientId.unique()
        
        for patient in patients:
            for date in dates:
                date_sub = data.loc[(data.GlucoseDisplayDate == date) & (data.PatientId == patient)]
                
                # obtain summary values
                median = np.nanmedian(date_sub.Value)
                mean = np.nanmean(date_sub.Value)
                std = np.nanstd(date_sub.Value)
                maxVal = np.nanmax(date_sub.Value)
                minVal = np.nanmin(date_sub.Value)
                
                # obtain areas above and below recommendation
                upperSegs = date_sub.loc[date_sub.Value > upper].Value - upper
                areaAbove = np.nansum(upperSegs)
                lowerSegs = -(date_sub.loc[date_sub.Value < lower].Value - lower)
                areaBelow = np.nansum(lowerSegs)
                
                sample = [date_sub.iloc[0]["PatientId"], median, mean, std, maxVal, minVal, areaBelow, areaAbove, date]
                
                median_df.loc[len(median_df.index)] = sample
        
        merged_median_df=data.merge(median_df, left_on=['PatientId', 'GlucoseDisplayDate'], right_on=['PatientId', 'Date'], how='left')
        merged_median_df=merged_median_df[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate',
                                           'inserted', 'missing', 'Median', 'Mean', 'Std Dev', 'Max', 'Min',
                                           'AreaBelow', 'AreaAbove']]

        return merged_median_df