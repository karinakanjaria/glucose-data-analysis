# Python Libraries
import numpy as np
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')
class Value_Imputation:
    '''preprocessing stuff'''
    def cleanup(self, subset: pd.DataFrame):
        '''replace 0s with NaN to save a two steps down the line'''
        subset['Value'] = subset['Value'].replace(0, np.nan)

        '''drop duplicate datetimes for each patient'''
        subset = subset.drop_duplicates(subset='GlucoseDisplayTime', keep="first")

        # ogNon0Values = subset['Value'].replace(0, np.nan).count()
        # print("starting with", ogNon0Values, "nonzero glucose measurements ('Value')")
        # print(subset.info())
        


        '''Find Unrecorded 5-minute Sample'''
        patID = subset['PatientId'].iloc[0]
        
        '''set datetime as index in order for resampling method below to work'''
        subset = subset.set_index('GlucoseDisplayTime')
        
        '''fill in missing timestamps'''
        subset = subset.resample('5min').first()

        '''fix columns that *need* to be filled in'''
        subset['PatientId'] = subset['PatientId'].replace(np.nan, patID)


        '''Interpolate Missing Data'''
        # Description goes here
        # 
        """
        subset['Value'] = subset['Value'].interpolate(method='pchip')
        missing_vals = subset[subset['missing'] == 1].index
        for i in missing_vals:
            lower_t = i - timedelta(hours = 5)
            upper_t = i - timedelta(hours = 3)
            std_prev = np.sqrt(np.std(subset.loc[lower_t:upper_t, 'Value']))
            jiggle = std_prev*np.random.randn()
            subset.loc[i, 'Value'] = subset.loc[i, 'Value'] + jiggle
        """
        temp = subset['Value'].mean()
        subset[subset.loc[subset['Value'].isna(), 'Value']] = temp
        
        subset=subset.reset_index(drop=False)
        
        subset=subset[['GlucoseDisplayTime', 'PatientId', 'Value']]
        
        return subset

    """
    def fill_missing_bootstrap(self, data):
        '''Keyword arguments:
        data -- DataFrame of all 10,000 patients'''
        data=data[data.duplicated(subset=['PatientId', 'GlucoseDisplayTime']) == False]
        conv_time = data.GlucoseDisplayTime.dt.time #save the display-time
        '''turning the time of day attribute into a float value where whole numbers are hours and fractions are minutes and seconds'''
        timeOfday = pd.Series(dtype=float)
        datedisplay = pd.Series(dtype=float)
        for i in conv_time.index:
            timeOfday.at[i] = conv_time[i].hour + conv_time[i].minute/60 + conv_time[i].second/3600
            datedisplay.at[i] = data.GlucoseDisplayTime[i].date()
        #timeOfday = cgm_functions.date2float(data.GlucoseDisplayTime.dt.time)
        data['GlucoseDisplayTimeNoDay'] = timeOfday
        data['GlucoseDisplayDate'] = datedisplay
        # '''Put It All Together'''
        # patIDs = data.PatientId.unique().tolist() #make list of patient IDs
        # for i in patIDs: #loop through all patient IDs
        #     interp = self.interpolateMissing(self.ZeroToNaN(self.fillUnrecorded(data[data.PatientId == i]))) #interpolate missing data for each patient
        #     data = data[data.PatientId != i] #delete the old data
        #     data = pd.concat([data,interp]) #save the new interpolated-filled data back into the dataframe
        '''Put It All Together'''
        # interp = self.interpolateMissing(self.ZeroToNaN(self.fillUnrecorded(data[data.PatientId == i]))) #interpolate missing data for each patient
        # data = data[data.PatientId != i] #delete the old data
        # data = pd.concat([data,interp]) #save the new interpolated-filled data back into the dataframe
        
        interp = self.interpolateMissing(self.ZeroToNaN(self.fillUnrecorded(data))) #interpolate missing data for each patient
        data = pd.concat([data,interp], ignore_index=True) #save the new interpolated-filled data back into the dataframe
        data=data[['PatientId', 'Value', 'GlucoseDisplayTime', 'GlucoseDisplayDate', 'inserted', 'missing']]
        return data
        """