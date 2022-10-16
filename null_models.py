"""
    Contains functions for calculating a continuous and binary null model likelihood.
"""
import sys
sys.path.insert(0, 'DMpy/')

import numpy as np

from DMpy import DMModel, Parameter
from DMpy.observation import softmax
from learning_BLT import rescorla_wagner

def continuous_null_likelihood(n_subjects, n_outcomes, data_path, fit_method):
    """
        Creates a continuous null model and calculates the null likelihood for each subject.
        Arguments:
            - n_subjects: number of subjects to calculate null likelihood for
            - n_outcomes: number of trial outcomes
            - data_path: filepath to subject data
            - fit_method: fit method to use
        Returns:
            - null_likelihoods: array of null likelihoods for each subject
    """
    # define model
    # initialise value
    value = Parameter('value', 'fixed', mean=0.5, dynamic=True)

    # initialise other learning parameters
    alpha = Parameter('alpha', 'fixed', mean=0.0, dynamic=False)

    #create model instance
    model = DMModel(rescorla_wagner, [value, alpha], None, None, logp_function='beta')

    #fit model
    model.fit(data_path, fit_method=fit_method, fit_stats=True, recovery=False)

    # get likelihood
    individual_fits = model.individual_fits(response_variable='value')
    null_likelihoods = np.empty(n_subjects, dtype=np.float64)

    for s in range(n_subjects):
        subject = s + 1
        trial_ll = individual_fits['logp'][(s*n_outcomes):(subject*n_outcomes)].to_numpy()
        trial_ll = np.where(trial_ll < -99999, 0, trial_ll) # deal with -inf values
        ll_len = len(trial_ll)
        null_likelihoods[s] = np.sum(trial_ll)

    return null_likelihoods

def binary_null_likelihood(n_subjects, n_outcomes, data_path, fit_method):
    """
        Creates a binary null model and calculates the null likelihood for each subject.
        Arguments:
            - n_subjects: number of subjects to calculate null likelihood for
            - n_outcomes: number of trial outcomes
            - data_path: filepath to subject data
            - fit_method: fit method to use
        Returns:
            - null_likelihoods: array of null likelihoods for each subject
    """
    # define model
    # initialise value
    value = Parameter('value', 'fixed', mean=0.5, dynamic=True)

    # initialise other parameters
    alpha = Parameter('alpha', 'fixed', mean=0.0, dynamic=False)
    beta = Parameter('beta', 'normal', mean=4.21, variance=1.75)

    #create model instance
    model = DMModel(rescorla_wagner,[value, alpha], softmax, [beta], logp_function='bernoulli')

    #fit model
    model.fit(data_path, fit_method=fit_method, fit_stats=True, recovery=False)

    # get likelihood
    individual_fits = model.individual_fits(response_variable='value')
    null_likelihoods = np.empty(n_subjects, dtype=np.float64)

    for s in range(n_subjects):
        subject = s + 1
        trial_ll = individual_fits['logp'][(s*n_outcomes):(subject*n_outcomes)].to_numpy()
        trial_ll = np.where(trial_ll < -99999, 0, trial_ll) # deal with -inf values
        ll_len = len(trial_ll)
        null_likelihoods[s] = np.sum(trial_ll)

    return null_likelihoods
