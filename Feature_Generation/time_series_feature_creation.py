import pandas as pd
import numpy as np
import fathon
from fathon import fathonUtils as fu
from fdasrsf import fPCA, time_warping
import EntropyHub as EH

class TS_Feature_Creation:
    def do_mfdfa(self, data: list, win_sizes: list, q_list: list, rev_seg: bool, pol_order: int):
        # zero-mean cumulative sum of data
        data_cs = fu.toAggregated(data)

        # init mfdfa object
        pymfdfa = fathon.MFDFA(data_cs)

        # compute fluctuation function and generalized Hurst exponents
        n, F = pymfdfa.computeFlucVec(win_sizes, q_list, rev_seg, pol_order)
        list_H, list_H_intercept = pymfdfa.fitFlucVec()

        # compute mass exponents
        tau = pymfdfa.computeMassExponents()

        # compute multifractal spectrum
        alpha, mfSpect = pymfdfa.computeMultifractalSpectrum()

        return n, F, list_H, list_H_intercept, tau, alpha, mfSpect


    def poincare_wrapper(self, data: pd.DataFrame):
        """
        INPUT: data	= cleaned DataFrame with verified unique patient-per-transmitter
        OUTPUT: poincare_df = DataFrame of 3xn where n is the number of patients, and the 3
                        columns are 'SD1', 'SD2', 'SD_ratio' (as definied by the func-
                        tion eclipseFittingMethod())
        """
        def eclipseFittingMethod(IDI): #function written by Jamie
            """
            Input: IDI = [list] of inter-data intervals (diff(time_series))
            Output: SD1, SD2 = {dict} with keys 'SD1' (numpy.float64), representing short-term
                        variation, and 'SD2' (numpy.float64), representing long-term
                        variation
            """
            SDSD = np.std(np.diff(IDI))
            SDRR = np.std(IDI)
            SD1 = (1 / np.sqrt(2)) * SDSD #measures the width of poincare cloud
            SD2 = np.sqrt((2 * SDRR ** 2) - (0.5 * SDSD ** 2)) #measures the length of the poincare cloud
            
            return {'SD1': round(SD1,3), 'SD2': round(SD2,3), 'SD_ratio': round(SD1/SD2,3)}

        poincare_df = pd.DataFrame(columns=['SD1', 'SD2', 'SD_ratio'])

        patIDs = data.PatientId.unique()
        for i in range(len(patIDs)):
            single_pat = data.loc[data.PatientId == patIDs[i]]

            glucose_diff = np.diff(single_pat.Value)
            poincare_df.loc[len(poincare_df)+1] = eclipseFittingMethod(glucose_diff)

        return poincare_df

    def fpcaWrapper(rawData, minTime, numComponents = 3):
	    #extracting columns 
	    dataNeeded = rawData[['PatientId', 'GlucoseDisplayTime', 'Value']]
	    dataNeeded = dataNeeded.astype({'PatientId' : "string",
		                            'GlucoseDisplayTime': 'datetime64[ns]',
		                            'Value' : 'int'})

	    dataNeeded['GlucoseTime'] = [time.strftime("%H:%M:%S") for time in dataNeeded.GlucoseDisplayTime]
	    dataNeeded['GlucoseDate'] = [time.date() for time in dataNeeded.GlucoseDisplayTime]

	    patientCoefs = []

	    patientIds = dataNeeded['PatientId'].unique()
	    days = dataNeeded['GlucoseDate'].unique()

	    for patientId in patientIds:      
		formattedData = dataNeeded[(dataNeeded['PatientId'] == patientId)] \
		    .pivot(index='GlucoseTime', columns='GlucoseDate', values='Value')

		#change to numpy to work with fxn
		glucoseValues = formattedData.to_numpy().astype(float)
		timeVec = np.linspace(0, 1, len(glucoseValues))

		#align time 
		fdaWarp = time_warping.fdawarp(glucoseValues, timeVec)
		fdaWarp.srsf_align()            

		# Run the FPCA, using Vertical fpca because we want to see the variation in value (y-axis)
		fpcaAnalysis = fPCA.fdavpca(fdaWarp)                                   
		fpcaAnalysis.calc_fpca(no=numComponents)                

		coefs = fpcaAnalysis.coef

		for index, coef in coef:
		    pcaFxn = fpcaAnalysis.f_pca[:,0,index]
		    patientCoefs.append([patientId, day, pcaIndex, coef, pcaFxn])

    return patientCoefs

    def removeDuplicatesByPatientAndTime(data):
    	return data[data.duplicated(subset=['PatientId', 'GlucoseDisplayTime']) == False]

    def entropy_calculation(self, data):
        ent_df=pd.DataFrame()
        patients=data['PatientId'].unique()

        for patient in patients:
            new_patient=data[data['PatientId']==patient]
            entropy=EH.SampEn(new_patient['Value'].values, m=4)[0][-1]
            ent_df['Entropy']=[entropy]

        return ent_df
