"""
    File name: utils_BLT.py
    Author: Katja Brand
    Created: 28/06/2021
    Description: This contains various functions used by main_BLT.py, models_BLT.py etc.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

def get_BLT_data(input_path, output_path, subID):
    """
        Converts matlab file and extracts outcomes and response data for model fitting.
        Arguments:
            input_path: the filepath to the matlab file containing the data
            output_path: the filepath where the .csv file will be stored
            subID: the subject ID corresponding to the data
        Returns:
            data: .csv file containing properly formatted data for model fitting
            outcomes: a numpy array containing the outcomes from the experiment
    """
    # get contents from matlab file
    contents = sio.loadmat(input_path)

    # extract outcomes/pairings
    m_params = contents['params'][0][0]
    outcomes = np.array(m_params[10]).reshape(len(m_params[10]))

    # extract subject responses
    m_data = contents['data'][0][0]
    responses = np.array(m_data[1]).reshape(len(m_data[1][0])) / 100

    subIDs = np.full((len(outcomes)), subID)

    # combine outcomes, responses, and subject ID into a dataframe, convert to csv file
    df = pd.DataFrame({'Outcome':outcomes, 'Response':responses, 'Subject':subIDs})
    data = df.to_csv(output_path, index=False)

    return data, outcomes

    #TODO: put outcomes, responses and subID into a .csv file, return file
