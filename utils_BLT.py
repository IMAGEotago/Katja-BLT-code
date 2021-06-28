"""
    File name: utils_BLT.py
    Author: Katja Brand
    Created: 28/06/2021
    Description: This contains various functions used by main_BLT.py, models_BLT.py etc.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

def get_BLT_data(filepath, subID):
    """
        Converts matlab file and extracts outcomes and response data for model fitting.
        Arguments:
            filepath: the filepath to the file containing the data
            subID: the subject ID corresponding to the data
        Returns:
            data: .csv file containing properly formatted data for model fitting
    """
    # get contents from matlab file
    contents = sio.loadmat(filepath)

    # extract outcomes/pairings
    m_params = contents['params'][0][0]
    outcomes = np.array(m_params[10]).reshape(len(m_params[10]))

    # extract subject responses
    m_data = contents['data'][0][0]
    responses = np.array(m_data[1]).reshape(len(m_data[1]))

    subIDs = np.full((len(outcomes),0), subID)

    df = pd.DataFrame('Outcome':outcomes, 'Response':responses, 'Subject':subIDs) #TODO: check this

    #TODO: put outcomes, responses and subID into a .csv file, return file
