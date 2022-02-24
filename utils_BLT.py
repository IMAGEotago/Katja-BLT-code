"""
    File name: utils_BLT.py
    Author: Katja Brand
    Created: 28/06/2021
    Description: This contains various functions used by main_BLT.py, models_BLT.py etc.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr

from scipy.stats.distributions import chi2

class Subject:
    """
        Represents each subject used for model fitting.
        Each subject has:
            id: subject ID number
            df: dataframe containing responses
            outcomes: an array of trial outcomes
            resist: log of resistance/no resistance for each trial
            sim_results: simulation results
    """
    def __init__(self, id, df, outcomes, resist, sim_results=None):
        self.id = id
        self.df = df
        self.outcomes = outcomes
        self.resist = resist
        self.sim_results = sim_results

def get_BLT_data(input_path, subID, continuous=True):
    """
        Converts matlab file and extracts outcomes and response data for model fitting.
        Arguments:
            input_path: the filepath to the matlab file containing the data
            subID: the subject ID corresponding to the data
            continuous: if False, data will be binarised (default is True)
        Returns:
            df: pandas dataframe containing the outcomes, responses, and subject ID
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

    # check responses matches outcomes length
    if len(responses) != len(outcomes):
        print(f"Warning: subject {subID} responses incomplete")
        n = np.empty(len(outcomes)-len(responses))
        n[:] = np.nan
        responses = np.concatenate((responses, n))

    # match responses to pairings
    for i in range(len(cues)):
        if cues[i] == 1:
            responses[i] = responses[i]
        else:
            responses[i] = 1 - responses[i]

    # check number of NaN responses
    if np.count_nonzero(np.isnan(responses)) > 0:
        print(f"Warning: subject {subID} has {np.count_nonzero(np.isnan(responses))} NaN response(s)")

    # binarise data if required
    if not continuous:
        for i in range(len(responses)):
            if responses[i] >= 0.5 and responses[i] <= 1.0:
                responses[i] = 1
            elif responses[i] < 0.5 and responses[i] >= 0.0:
                responses[i] = 0
            else:
                responses[i] = np.nan
    else:
        for i in range(len(responses)):
            if responses[i] == 9.99 or responses[i] == -8.99:
                responses[i] = 0.5

    subIDs = np.full((len(outcomes)), subID)

    # resistances stored in resist
    resist = np.array(m_params[7]).reshape(len(m_params[7]))

    # combine outcomes, responses, and subject ID into a dataframe, convert to csv file
    df = pd.DataFrame({'Outcome':outcomes, 'Response':responses, 'Subject':subIDs, 'Resistance':resist})

    return df, outcomes, resist

def get_model_stats(model, n_subjects, n_outcomes, continuous):
    """
        Calculates statistics for each subject including:
            - Model log likelihood
            - Likelihood ratio (and associated p-value) for model compared to chance
            - Pseudo-r2 value
        Arguments:
            model: the model to calculate the stats for
            n_subjects: number of subjects
            n_outcomes: number of outcomes/trials
            continuous: whether or not the model is continuous
    """
    # get fit data from model
    if continuous:
        individual_fits = model.individual_fits(response_variable='value')
    else:
        individual_fits = model.individual_fits(response_variable='prob')

    # calculate and print stats
    for s in range(n_subjects):
        subject = s + 1
        trial_ll = individual_fits['logp'][(s*n_outcomes):(subject*n_outcomes)].to_numpy()
        trial_ll = np.where(trial_ll < -99999, 0, trial_ll) # deal with -inf values
        ll_len = len(trial_ll)
        log_likelihood = np.sum(trial_ll)

        # likelihood ratio test
        if continuous:
            lr, p = likelihood_ratio(ll_len*np.log(1/101), log_likelihood)
        else:
            lr, p = likelihood_ratio(ll_len*np.log(0.5), log_likelihood)
        print(f"\nSubject {subject}")
        print(f"Model log likelihood: {log_likelihood}")
        print(f"likelihood ratio: {lr}")
        print("p: %.30f" %p)
        print(f"pseudo-r2 = {1 - (log_likelihood / (ll_len*np.log(0.5)))}")

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

def get_certainty(id, df):
    """
        Calculates the average certainty for a given participant.
        Arguments:
            id: Participant id
            df: Dataframe containing participant responses
    """
    responses = df["Response"].values
    certainty = 0.0

    for r in responses:
        if not math.isnan(r):
            c = abs(r - 0.5)
            certainty = certainty + c

    certainty = certainty / len(responses)
    print(f"Average certainty for participant {id} was {certainty}")

def get_proportion_correct(subjects, n_trials, plot):
    """
        Calculates proportion correct and average certainty for each trial across all subjects.
        Arguments:
            subjects: list of subjects
            n_trials: number of trials
            plot: if true, creates plots
        Returns:
            proportions_df: dataframe containing proportion correct and average certainty for each trial
    """
    proportions = np.zeros(n_trials)
    certainty = np.zeros(n_trials)
    for s in subjects:
        for i in range(n_trials):
            o = s.df['Outcome'][i]
            r = s.df['Response'][i]
            if (o == 1 and r > 0.5) or (o == 0 and r < 0.5):
                proportions[i] = proportions[i] + 1

            c = abs(r - 0.5)
            if not math.isnan(r):
                certainty[i] = certainty[i] + c

    proportions = proportions / len(subjects)
    certainty = certainty / len(subjects)
    trials = np.arange(n_trials) + 1

    proportions_df = pd.DataFrame({'Trial':trials, 'Proportion':proportions, 'Certainty':certainty})

    if plot == True:
        # get neuron data
        neuron_df = pd.read_csv('input_files/NeuronDataProportionCorrect.csv', header=0, index_col=0)
        neuron_proportions = neuron_df['PROPORTION CORRECT'].values

        r_val, p_val = pearsonr(proportions, neuron_proportions)
        print(f"Pearson's r: {r_val}                p value: {p_val}")

        plt.figure(figsize=(15,4))
        plt.plot(trials, proportions, color='tab:red', label='Current study')
        plt.plot(trials, neuron_proportions, color='k', label='Previous study')
        plt.legend(loc=4)
        plt.xlabel('Trials')
        plt.text(77, 0.95, f'R = {np.round(r_val,4)}', fontsize='large')
        plt.show()

    return proportions_df
