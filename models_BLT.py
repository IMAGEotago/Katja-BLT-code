"""
    File name: models_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains functions used to create and use DMpy models for main_BLT.py
"""
import matplotlib.pyplot as plt
import numpy as np

from DMpy import DMModel, Parameter
from DMpy.learning import rescorla_wagner
from DMpy.observation import softmax

from params import n_subjects, outcomes, n_outcomes, sim_path, sim_noise, fit_method, data_path


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
                                   output_file=sim_path,
                                   learning_parameters={'value' : values["value"],
                                                        'alpha' : values["alpha"]},
                                   noise_sd=sim_noise,
                                   return_choices=False,
                                   response_variable='value')
    else:
        _, sim_rw = model.simulate(outcomes=outcomes,
                                   n_subjects=n_subjects,
                                   output_file=sim_path,
                                   learning_parameters={'value' : values["value"],
                                                        'alpha' : values["alpha"]},
                                   observation_parameters={'beta' : values["beta"]},
                                   noise_sd=sim_noise,
                                   return_choices=True,
                                   response_variable='value')

    # Plot results from simulation
    if sim_plot:
        x = np.arange(n_outcomes)

        plt.figure(figsize=(15,3))

        # Plot results for each subject
        a = np.zeros(n_subjects)
        for i in range(n_subjects):
            n = i*n_outcomes
            a[i] = model.simulation_results['alpha_sim'][n]
            plt.plot(x, model.simulation_results['value'][n:(n + n_outcomes)], c=plt.cm.coolwarm(a[i]/3), alpha=0.5)

        plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
        plt.title(f'Simulation results for {n_subjects} subjects')
        plt.xlabel('Trial')
        plt.ylabel('Estimated value')
        plt.show()

    # Perform parameter recovery
    if recover:
        model.fit(sim_rw, fit_method=fit_method, fit_stats=True, recovery=True)

def fit_model(model, continuous=False, plot=True):
    """
        Fits the model to real data, and provides a simulated plot of the fitted alpha value/s.
        Arguments:
            model: the model to be used for fitting the data
            continuous: whether model is continuous (True) or binary (False)
            plot: if True, will plot the simulated behaviour from the fitted alpha value/s
    """
    # Fits the data
    model.fit(data_path, fit_method=fit_method, fit_stats=True, recovery=False)

    # Plots the fitted alpha values
    if plot:
        alpha = model.parameter_table["alpha"]
        n_alpha = len(alpha)

        # Simulate with fitted alpha values
        # TODO: this part is still buggy
        if continuous:
            _, sim_a = model.simulate(outcomes=outcomes,
                                      n_subjects=n_alpha,
                                      output_file='output_files/sim_fit_responses', #TODO: move to params?
                                      learning_parameters={'value' : [0.5]*n_alpha,
                                                           'alpha' : alpha},
                                      return_choices=False,
                                      response_variable='value')
        else:
            _, sim_a = model.simulate(outcomes=outcomes,
                                      n_subjects=n_alpha,
                                      output_file='output_files/sim_fit_responses', #TODO: move to params?
                                      learning_parameters={'value' : [0.5]*n_alpha,
                                                           'alpha' : alpha},
                                      observation_parameters={'beta' : [1.5]*n_alpha}, #TODO: get beta from params?
                                      return_choices=True,
                                      response_variable='value')

        x = np.arange(n_outcomes)
        plt.figure(figsize=(15,3))

        # Plot simulated behaviour for each subject
        a = np.zeros(n_alpha)
        for i in range(n_alpha):
            n = i*n_outcomes
            a[i] = model.simulation_results['alpha_sim'][n]
            plt.plot(x, model.simulation_results['value'][n:(n + n_outcomes)], c=plt.cm.twilight(a[i]/3), alpha=0.5,
                     label='alpha = %.3f' %a[i])

        plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
        plt.title(f'Simulated behaviour for fitted alpha values')
        plt.xlabel('Trial')
        plt.ylabel('Estimated value')
        plt.show()
