"""
    File name: functions_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains functions used to create and use models for main_BLT.py
"""

import numpy as np
from params import n_subjects
from DMpy import DMModel, Parameter
from DMpy.learning import rescorla_wagner
from DMpy.observation import softmax

def define_model(continuous=False):
    """
        Initialises parameters and defines the model.
        Arguments:
            continuous: determines whether a binary or contiuous response model is created
            Returns:
            model: the initialised model
    """
    # initialise parameters
    # TODO: adjust so means/bounds are not hard-coded
    value = Parameter('value', 'fixed', mean=0.5, dynamic=True)
    alpha = Parameter('alpha', 'uniform', lower_bound=0.0, upper_bound=1.0)

    value_values = [0.5] * n_subjects
    alpha_values = np.random.uniform(0.0, 1.0, n_subjects)

    if not continuous:
        beta = Parameter('beta', 'flat', mean=1.5)
        beta_values = [1.5] * n_subjects

    # create model instance
    if continuous:
        model = DMModel(rescorla_wagner,[value, alpha], None, None, logp_function='beta')
    else:
        model = DMModel(rescorla_wagner,[value, alpha], softmax, [beta], logp_function='bernoulli')

    return model


#TODO: add function for model simulation
