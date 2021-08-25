"""
    File name: utils_BLT.py
    Author: Katja Brand
    Created: 28/06/2021
    Description: This contains various functions used by main_BLT.py, models_BLT.py etc.
"""
import numpy as np
import pandas as pd
import scipy.io as sio

from scipy.stats.distributions import chi2

def get_BLT_data(input_path, output_path, subID, continuous=True):
    """
        Converts matlab file and extracts outcomes and response data for model fitting.
        Arguments:
            input_path: the filepath to the matlab file containing the data
            output_path: the filepath where the .csv file will be stored
            subID: the subject ID corresponding to the data
            continuous: if False, data will be binarised (default is True)
        Returns:
            outcomes: a numpy array containing the outcomes from the experiment
    """
    # get contents from matlab file
    contents = sio.loadmat(input_path)

    # extract outcomes/pairings
    m_params = contents['params'][0][0]
    outcomes = np.array(m_params[10]).reshape(len(m_params[10]))

    # extract cues
    cues = np.array(m_params[9]).reshape(len(m_params[9]))

    # extract subject responses
    m_data = contents['data'][0][0]
    responses = np.array(m_data[1]).reshape(len(m_data[1][0])) / 100

    # match responses to pairings
    for i in range(len(cues)):
        if cues[i] == 1:
            responses[i] = responses[i]
        else:
            responses[i] = 1 - responses[i]

    # binarise data if required
    if not continuous:
        for i in range(len(responses)):
            if responses[i] >= 0.5:
                responses[i] = 1
            else:
                responses[i] = 0

    subIDs = np.full((len(outcomes)), subID)

    # combine outcomes, responses, and subject ID into a dataframe, convert to csv file
    df = pd.DataFrame({'Outcome':outcomes, 'Response':responses, 'Subject':subIDs})
    df.to_csv(output_path, index=False)

    return outcomes

def likelihood_ratio(llmin, llmax):
    """
        Calculates the likelihood ratio given the log likelihoods, and subsequent p-value.
        Arguments:
            llmin: log likelihood #1
            llmax: log likelihood #2
        Returns:
            lr: the likelihood ratio
            p: the p-value of the likelihood ratio 
    """
    lr = 2*(llmax-llmin)
    p = chi2.sf(lr, 1)
    return lr, p
