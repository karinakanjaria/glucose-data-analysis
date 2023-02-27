import pandas as pd
import numpy as np
import fathon
from fathon import fathonUtils
import EntropyHub as eH


class TS_Feature_Creation:
    """
    INPUT: data	= Pandas DataFrame with `Value` column representing patient glucose values and `PatientID` values
    OUTPUT: featured_data = Pandas DataFrame with `PatientID` and Features generated from each

    Class with methods to extract features from time series data.
    """

    def mfdfa_extraction(self, data: pd.DataFrame, win_sizes: list, q_list: list, rev_seg: bool, pol_order: int):
        """
        mfdfa_extraction
        INPUT: data	= Pandas DataFrame with 'PatientID' and 'Value' columns
                    representing unique Pateints and their Glucose Values
        OUTPUT: mfdfa_features = Pandas DataFrame with 'PatientID', 'nWinSize', 'fWinFluct', 'hSlope', 'hIntercept',
                    'tau', 'alpha', 'mfSpect' arrays
        Performs Multi-Fractal Detrended Fluctuation Analysis on data to extract features
        """
        mfdfa_features = pd.DataFrame(
            columns=['PatientID', 'nWinSize', 'fWinFluct', 'hSlope', 'hIntercept', 'tau', 'alpha', 'mfSpect'])

        patients = data['PatientId'].unique()
        for patient in patients:
            new_patient = data[data['PatientId'] == patient]
            new_patient_values = new_patient['Value'].values

            # Finds zero-mean cumulative sum of data
            data_cs = fathonUtils.toAggregated(new_patient_values)

            # Initilize the mfdfa object
            pymfdfa = fathon.MFDFA(data_cs)

            # compute fluctuation function and generalized Hurst exponents
            n_window_sizes, f_window_fluctuations = pymfdfa.computeFlucVec(win_sizes, q_list, rev_seg, pol_order)
            h_slopes, h_intercept = pymfdfa.fitFlucVec()

            # compute mass exponents
            tau = pymfdfa.computeMassExponents()

            # compute multifractal spectrum
            alpha_singularity_strengths, mf_spect = pymfdfa.computeMultifractalSpectrum()

            patient_features = pd.DataFrame([[patient, n_window_sizes, f_window_fluctuations, h_slopes, h_intercept,
                                              tau, alpha_singularity_strengths, mf_spect]],
                                            columns=['PatientID', 'nWinSize', 'fWinFluct', 'hSlope', 'hIntercept',
                                                     'tau', 'alpha', 'mfSpect'])

            mfdfa_features = pd.concat([mfdfa_features, patient_features], ignore_index=True)
        return mfdfa_features

    def poincare_extraction(self, data: pd.DataFrame):
        """
        INPUT: data	= Pandas DataFrame with 'PatientID' and 'Value' columns
                    representing unique Patients and their Glucose Values
        OUTPUT: poincare_features = DataFrame of 3xn where n is the number of patients, and the 3
                    columns are 'shortTermVar', 'longTermVar', 'varRatio')
        """
        poincare_features = pd.DataFrame(columns=['PatientID', 'shortTermVar', 'longTermVar', 'varRatio'])
        pids = data.PatientId.unique()
        for i in range(len(pids)):
            patient = data.loc[data.PatientId == pids[i]]

            # Find inter-data differentials
            glucose_differentials = np.diff(patient.Value)

            st_dev_differentials = np.std(np.diff(glucose_differentials))
            st_dev_values = np.std(glucose_differentials)

            # measures the width of poincare cloud
            short_term_variation = (1 / np.sqrt(2)) * st_dev_differentials

            # measures the length of the poincare cloud
            long_term_variation = np.sqrt((2 * st_dev_values ** 2) - (0.5 * st_dev_differentials ** 2))

            poincare_features.loc[len(poincare_features) + 1] = [patient,
                                                                 round(short_term_variation, 3),
                                                                 round(long_term_variation, 3),
                                                                 round(short_term_variation / long_term_variation, 3)]
        return poincare_features

    def entropy_extraction(self, data):
        """
        INPUT: data	= Pandas DataFrame with 'PatientID' and 'Value' columns
                    representing unique Patients and their Glucose Values
        OUTPUT: entropy_features = DataFrame with 'PatientID' and 'Entropy' values
        """
        entropy_features = pd.DataFrame(columns=['PatientID', 'Entropy'])
        patients = data['PatientId'].unique()

        for patient in patients:
            new_patient = data[data['PatientId'] == patient]
            entropy = eH.SampEn(new_patient['Value'].values, m=4)[0][-1]
            patient_features = pd.DataFrame([[patient, entropy]], columns=['PatientID', 'Entropy'])
            entropy_features = pd.concat([entropy_features, patient_features], ignore_index=True)
        return entropy_features
