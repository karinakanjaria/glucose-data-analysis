# Python Libraries
import numpy as np
import pandas as pd


def compressDailyValues(data):
    '''Keyword arguments:
    data -- DataFrame of a single patient with no missing data'''

    fs = 1/300
    lower = 70
    upper = 180
    
    median_df = pd.DataFrame(columns=['PatientId', 'Median', 'Mean', 'Std Dev', 'Max', 'Min', 'AreaBelow', 'AreaAbove', 'Date'])
    dates = data.GlucoseDisplayDate.unique()
    
    for date in dates:
        date_sub = data.loc[data.GlucoseDisplayDate == date]
        
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
        
        sample = [data.iloc[0]["PatientId"], median, mean, std, maxVal, minVal, areaBelow, areaAbove, date]
        
        median_df.loc[len(median_df.index)] = sample
                                   
    return median_df