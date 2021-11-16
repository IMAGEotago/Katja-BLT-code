"""
    File name: models_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains functions used to create and use DMpy models for main_BLT.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from DMpy import DMModel, Parameter
from DMpy.observation import softmax

from params import n_subjects, outcomes, n_outcomes, sim_path, sim_noise, data_path, fit_method, beta_val, subjects
from learning_BLT import rescorla_wagner, dual_lr_rw


def define_model(model_type=rescorla_wagner, continuous=False):
    """
        Initialises parameters and defines a model.
        Arguments:
            model_type: the type of learning model to be used (default Rescorla-Wagner).
            continuous: determines whether a binary or contiuous response model is created (default binary)
        Returns:
            model: the initialised model
            values: dictionary containing the simulation values for the initialised parameters
    """
    # initialise value parameter
    value = Parameter('value', 'fixed', mean=0.5, dynamic=True)

    # save learning parameter values in dict
    l_values = {}
    l_values["value"] = [0.5] * n_subjects

    # initialise other learning parameters
    if model_type == rescorla_wagner:
        if fit_method == 'MLE':
            alpha = Parameter('alpha', 'uniform', lower_bound=0.0, upper_bound=1.0)
        else:
            alpha = Parameter('alpha', 'normal', lower_bound=0.0, upper_bound=1.0, mean=0.29, variance=2.54)
        l_values["alpha"] = np.random.uniform(0.0, 1.0, n_subjects) #TODO: match priors?

    if model_type == dual_lr_rw:
        alpha_n = Parameter('alpha_n', 'normal', lower_bound=0.0, upper_bound=1.0, mean=0.29, variance=2.54)
        alpha_p = Parameter('alpha_p', 'normal', lower_bound=0.0, upper_bound=1.0, mean=0.29, variance=2.54)
        l_values["alpha_p"] = np.random.uniform(0.0, 1.0, n_subjects)
        l_values["alpha_n"] = np.random.uniform(0.0, 1.0, n_subjects)

    # if binary model, create and save observation parameter/s
    if not continuous:
        if fit_method == 'MLE':
            beta = Parameter('beta', 'flat')
        else:
            beta = Parameter('beta', 'normal', mean=2.14, variance=3.33)
        o_values = {}
        o_values["beta"] = [beta_val] * n_subjects

    # create model instance
    if model_type == rescorla_wagner:
        if continuous:
            model = DMModel(rescorla_wagner,[value, alpha], None, None, logp_function='beta')
        else:
            model = DMModel(rescorla_wagner,[value, alpha], softmax, [beta], logp_function='bernoulli')
    elif model_type == dual_lr_rw:
        if continuous:
            model = DMModel(dual_lr_rw,[value, alpha_p, alpha_n], None, None, logp_function='beta')
        else:
            model = DMModel(dual_lr_rw,[value, alpha_p, alpha_n], softmax, [beta], logp_function='bernoulli')
    else:
        print(f"Error: model_type {model_type} not recognised.")

    return model, l_values, o_values


def model_simulation(model, l_values, o_values, continuous=False, sim_plot=True, recover=True):
    """
        Simulates from model, and performs parameter recovery if necessary
        Arguments:
            model: the model to simulate from
            values: parameter values of the model
            continuous: determines whether a binary or contiuous response model is created (default binary)
            sim_plot: if True, will produce a plot of the simulated results
            recover: if True, parameter recovery will be performed (default True)
    """

    # Run simulation on the model
    if continuous:
        if model.learning_model == dual_lr_rw:
            _, sim_rw = model.simulate(outcomes=outcomes,
                                       n_subjects=n_subjects,
                                       output_file=sim_path,
                                       learning_parameters=l_values,
                                       noise_sd=sim_noise,
                                       return_choices=False,
                                       combinations=False,
                                       response_variable='value',
                                       model_inputs={'Resistance'})
        else:
            _, sim_rw = model.simulate(outcomes=outcomes,
                                       n_subjects=n_subjects,
                                       output_file=sim_path,
                                       learning_parameters=l_values,
                                       noise_sd=sim_noise,
                                       return_choices=False,
                                       combinations=False,
                                       response_variable='value')
    else:
        if model.learning_model == dual_lr_rw:
            print("Running simulation using dual_lr_rw")
            _, sim_rw = model.simulate(outcomes=outcomes,
                                       n_subjects=n_subjects,
                                       output_file=sim_path,
                                       learning_parameters=l_values,
                                       observation_parameters=o_values,
                                       return_choices=True,
                                       combinations=False,
                                       response_variable='prob',
                                       model_inputs={'Resistance'})
        else:
            _, sim_rw = model.simulate(outcomes=outcomes,
                                       n_subjects=n_subjects,
                                       output_file=sim_path,
                                       learning_parameters=l_values,
                                       observation_parameters=o_values,
                                       return_choices=True,
                                       combinations=False,
                                       response_variable='prob')

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
        model.fit(sim_rw, fit_method=fit_method, fit_stats=True, recovery=True, suppress_table=False)

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
        alpha_vals = np.array(model.parameter_table["alpha"])

        # make sure there are at least 2 alpha values
        if len(alpha_vals) == 1:
            alpha_vals = np.tile(alpha_vals, 2)
        n_alpha = len(alpha_vals)

        # Simulate with fitted parameter values
        # TODO: see if plot_against_true can be debugged?
        if continuous:
            _, sim_a = model.simulate(outcomes=outcomes,
                                      n_subjects=n_alpha,
                                      return_choices=False,
                                      response_variable='value')
        else:
            beta_vals = np.array(model.parameter_table["beta"])
            _, sim_a = model.simulate(outcomes=outcomes,
                                      n_subjects=n_alpha,
                                      return_choices=True,
                                      response_variable='prob')

        x = np.arange(n_outcomes)
        plt.figure(figsize=(15,3))

        # Plot simulated behaviour for each subject
        a = np.zeros(n_alpha)
        for i in range(n_alpha):
            n = i*n_outcomes
            a[i] = model.simulation_results['alpha_sim'][n]
            subjects[i].sim_results = model.simulation_results['value'][n:(n + n_outcomes)]
            plt.plot(x, model.simulation_results['value'][n:(n + n_outcomes)], c=plt.cm.plasma(a[i]), alpha=0.5,
                     label=f"Subject {subjects[i].id};    alpha = {np.round(a[i],3)}")

        plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
        plt.title(f"Simulated behaviour for fitted alpha values for {len(subjects)} participants")
        plt.legend()
        plt.xlabel('Trial')
        plt.ylabel('Estimated value')
        plt.show()

def plot_trajectories(s):
    """
        Plots predictions and outcomes for given subject.
        Arguments:
            s: subject to be plotted
    """

    predictions = s.df.loc[:,'Response']
    state = [0.8] * 30 + [0.2] * 12 + [0.8] * 13 + [0.2] * 12 + [0.8] * 13

    # correlation between participant predictions and simulated Predictions
    predictions = np.nan_to_num(predictions, nan=0.5)
    r_val, p_val = pearsonr(predictions, s.sim_results)

    print(f"Plotting trajectory for participant {s.id}")
    print(f"R = {np.round(r_val, 6)}, p = {np.round(p_val, 6)}")

    plt.figure(figsize=(15,3))
    plt.plot(s.outcomes, 'o', state, '-', c='darkred', alpha=0.8)
    plt.plot(predictions, '-', c='tab:red', label="Predictions")
    plt.plot(np.arange(n_outcomes), s.sim_results, '-', c='coral', label="Simulated trajectory")
    plt.title(f"Prediction trajectory for participant {s.id}")
    plt.legend()
    plt.show()
