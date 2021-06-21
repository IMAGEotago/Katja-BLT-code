"""
    File name: functions_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains functions used to create and use models for main_BLT.py
"""
import matplotlib.pyplot as plt
import numpy as np

from DMpy import DMModel, Parameter
from DMpy.learning import rescorla_wagner
from DMpy.observation import softmax

from params import n_subjects, outcomes, n_outcomes


def define_model(continuous=False):
    """
        Initialises parameters and defines the model.
        Arguments:
            continuous: determines whether a binary or contiuous response model is created (default binary)
        Returns:
            model: the initialised model
            values: dictionary containing the values for the initialised parameters
    """
    # initialise parameters
    # TODO: adjust so means/bounds are not hard-coded
    value = Parameter('value', 'fixed', mean=0.5, dynamic=True)
    alpha = Parameter('alpha', 'uniform', lower_bound=0.0, upper_bound=1.0)

    # save parameter values in dict
    values = {}
    values["value"] = [0.5] * n_subjects
    values["alpha"] = np.random.uniform(0.0, 1.0, n_subjects)

    if not continuous:
        beta = Parameter('beta', 'flat', mean=1.5)
        values["beta"] = [1.5] * n_subjects

    # create model instance
    if continuous:
        model = DMModel(rescorla_wagner,[value, alpha], None, None, logp_function='beta')
    else:
        model = DMModel(rescorla_wagner,[value, alpha], softmax, [beta], logp_function='bernoulli')

    return model, values


def model_simulation(model, values, continuous=False, sim_plot=True, recover=True):
    """
        Simulates from model, and performs parameter recovery if necessary
        Arguments:
            model: the model to simulate from
            values: parameter values of the model
            continuous: determines whether a binary or contiuous response model is created (default binary)
            sim_plot: if True, will produce a plot of the simulated results
            recover: if True, parameter recovery will be performed (default True)
    """
    # Call simulate() function with relevant parameters
    if continuous:
        _, sim_rw = model.simulate(outcomes=outcomes,
                                   n_subjects=n_subjects,
                                   output_file='output_files/test_blt_responses.csv',
                                   learning_parameters={'value' : values["value"],
                                                        'alpha' : values["alpha"]},
                                   noise_sd=0.0, #TODO: add params variable for noise
                                   return_choices=False,
                                   response_variable='value')
    else:
        _, sim_rw = model.simulate(outcomes=outcomes,
                                   n_subjects=n_subjects,
                                   output_file='output_files/test_blt_responses.csv',
                                   learning_parameters={'value' : values["value"],
                                                        'alpha' : values["alpha"]},
                                   observation_parameters={'beta' : values["beta"]},
                                   noise_sd=0.0, #TODO: add params variable for noise
                                   return_choices=True,
                                   response_variable='value')

    # Plot results from simulation
    if sim_plot:
        x = np.arange(n_outcomes)

        a1 = model.simulation_results['alpha_sim'][0]
        a2 = model.simulation_results['alpha_sim'][n_outcomes]
        a3 = model.simulation_results['alpha_sim'][2*n_outcomes]

        plt.figure(figsize=(15, 3))
        plt.plot(x, model.simulation_results['value'][0:n_outcomes], label='alpha_s1 = %.3f' %a1)
        plt.plot(x, model.simulation_results['value'][n_outcomes:n_outcomes*2], label='alpha_s2 = %.3f' %a2)
        plt.plot(x, model.simulation_results['value'][n_outcomes*2:n_outcomes*3], label='alpha_s3 = %.3f' %a3)
        plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Estimated value')
        plt.show()

    # Perform parameter recovery
    if recover:
        model.fit(sim_rw, fit_method='MLE', fit_stats=True, recovery=True) #TODO: get fit_method from params
