# Python Libraries
import numpy as np
import pandas as pd

# import warnings
# warnings.filterwarnings('ignore')
class Value_Imputation:
    '''Insert Row into Dataframe'''
    def insertRow(self, row_number, df, row_value):
        start_upper = 0 # Starting value of upper half
        end_upper = row_number # End value of upper half
        start_lower = row_number # Start value of lower half
        end_lower = df.shape[0] # End value of lower half
        upper_half = [*range(start_upper, end_upper, 1)] # Create a list of upper_half index
        lower_half = [*range(start_lower, end_lower, 1)] # Create a list of lower_half index
        lower_half = [x.__add__(1) for x in lower_half] # Increment the value of lower half by 1
        index_ = upper_half + lower_half # Combine the two lists
        df.index = index_ # Update the index of the dataframe
        df.loc[row_number] = row_value # Insert a row at the end
        df = df.sort_index() # Sort the index labels
        return df


    '''Find Unrecorded 5-minute Sample'''
    def fillUnrecorded(self, df):
        # Description: Find gaps in sampling and insert samples that fill the time-gap with a missing value
        #     INPUTS: 
        #         df: dataframe of the raw data that hasn't replaced zeros with NaNs
        #     OUTPUTS:
        #         df_filled: dataframe with inserted timestamps
        
        df['inserted'] = 0
        stitched_df = pd.DataFrame(columns = df.columns)

        df = df.sort_values("GlucoseDisplayTime", ascending=True) \
            .reset_index(drop=True) #bit of cleanup
        display_times = df.GlucoseDisplayTime
        
        for i in range(1, len(display_times)-1):
            delta_sec = (display_times[i] - display_times[i-1]).seconds
            steps_missing = int(np.round(delta_sec / 300)-1) # how many 5-minute samples are missing #may come back to clean

            if steps_missing > 0:
                for step in range(steps_missing):
                    row_number = i + (step-1)
                    row_data = df.iloc[i-1]
                    time_feats = ['RecordedSystemTime', 'RecordedDisplayTime', 'GlucoseSystemTime', 'GlucoseDisplayTime']
                    row_data[time_feats] = row_data[time_feats] + np.timedelta64(5*(step+1), 'm')
                    row_data['inserted'] = 1
                    subset_fill = self.insertRow(row_number, df, row_data)
            else:
                subset_fill = df
        stitched_df = pd.concat([stitched_df, subset_fill])
        stitched_df.sort_values("GlucoseDisplayTime", ascending=True, inplace=True)
        stitched_df.reset_index(drop=True, inplace=True)
        return stitched_df


    '''Convert Zero Values to NaN'''
    def ZeroToNaN(self, df):
        # Description: Replace the zero values in the data with missing (NaN) values for interpolation methods
        #     INPUTS
        #        df: dataframe of raw data that has incorrectly labeled zero values for CGM values
        #     OUTPUTS
        #        df_missing: dataframe with zeros converted to NaNs
        
        df['missing'] = pd.Series(0, df.index)
        df.Value.replace(0, np.nan, inplace=True)

        missing_label = df[df.Value.isnull()].index
        df['missing'].iloc[missing_label] = 1
            
        return df


    '''Interpolate Missing Data'''
    def interpolateMissing(self, df):
        # Description goes here
        # 
        df['missing'] = pd.Series(0, df.index)
        df.Value = df.Value.replace(0, np.nan)
        missing_label = df[df.Value.isnull()].index
        df['missing'].iloc[missing_label] = 1
        
        df = df.sort_values("GlucoseDisplayTime", ascending=True) \
            .reset_index(drop=True)
        df.Value = df.Value.interpolate(method='pchip')
        
        missing_vals = df.loc[df.missing == 1]
        sigma = np.sqrt(np.std(df.Value))
        
        for i in missing_vals.index:
            jiggle = sigma*np.random.randn()
            df.at[i, 'Value'] = df.at[i, 'Value'] + jiggle
            
        return df.dropna(axis=0, subset="Value")

    
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