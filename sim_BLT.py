"""
    File name: sim_BLT.py
    Author: Katja Brand
    Created: 25/08/2021
    Description: This is a script for running multiple model simulations and
                 saving the results.
"""
import sys
sys.path.insert(0, 'DMpy/')

import numpy as np
import matplotlib.pyplot as plt
import os

import params
from learning_BLT import rescorla_wagner, dual_lr_rw
from models_BLT import define_model, model_simulation

# get current file path
fileDir = os.path.dirname(os.path.realpath('__file__'))

# repeat each run 10 times
repeat = 10
run = 1

# number of plots produced in one run
n_plots = 4

while run <= repeat:
    # checkpoint
    if params.continuous:
        print(f"Continuous model run {run} with noise {params.sim_noise} using the {params.model_type} learning model")
    else:
        print(f"Binary model run {run} with beta {params.beta_val} using the {params.model_type} learning model")

    # save all output to text file
    file_path = ""
    n = int(params.sim_noise*100)
    if params.model_type == dual_lr_rw:
        file_path = os.path.join(fileDir, "../sim_results/dual_lr_rw/")
    else:
        file_path = os.path.join(fileDir, "../sim_results/")

    if params.continuous:
        file_path = os.path.join(file_path, f"continuous/noise_{n}/run_{run}")
        text_path = os.path.join(file_path, f"con_n{n}_log.txt")
        noise_path = os.path.join(file_path, f"con_n{n}_noise.txt")
    else:
        file_path = os.path.join(file_path, f"binary/beta_{params.beta_val}/run_{run}")
        text_path = os.path.join(file_path, f"bin_b{params.beta_val}_log.txt")
        noise_path = None

    default_stdout = sys.stdout
    sys.stdout = open(text_path, "w")

    # print relevant model parameters
    if params.continuous:
        print(f"Running continuous model with {params.n_subjects} subjects and {params.n_outcomes} trials using the {params.model_type} learning model")
        print(f"Noise SD: {params.sim_noise}")
    else:
        print(f"Running binary model with {params.n_subjects} subjects and {params.n_outcomes} trials using the {params.model_type} learning model")
        print(f"Beta: {params.beta_val}")

    # create model
    model, l_values, o_values = define_model(model_type=params.model_type, continuous=params.continuous)

    # run simulation and parameter recovery
    model_simulation(model, l_values, o_values, continuous=params.continuous, recover=True, sim_plot=True, noise_path=noise_path)

    # save plots
    figs = [plt.figure(n) for n in plt.get_fignums()[-n_plots:]]
    i = 1
    for fig in figs:
        if params.continuous:
            fig_name = f"con_n{n}_f{i}"
        else:
            fig_name = f"bin_b{params.beta_val}_f{i}"

        fig_path = os.path.join(file_path, f"{fig_name}.png")
        fig.savefig(fig_path)
        i = i+1

    # get alpha and beta values
    if not params.continuous:
        beta_column = model.parameter_table.loc[:,"beta"]
        beta_values = beta_column.values
        log_beta_values = np.log10(beta_values)

    if params.model_type == dual_lr_rw:
        # alpha_p
        alpha_column = model.parameter_table.loc[:,"alpha_p"]
        alpha_values = alpha_column.values

        alpha_sim_col = model.parameter_table.loc[:,"alpha_p_sim"]
        alpha_sim_vals = alpha_sim_col.values

        # alpha_n
        alpha_n_column = model.parameter_table.loc[:,"alpha_n"]
        alpha_n_values = alpha_n_column.values

        alpha_n_sim_col = model.parameter_table.loc[:,"alpha_n_sim"]
        alpha_n_sim_vals = alpha_n_sim_col.values
    else:
        alpha_column = model.parameter_table.loc[:,"alpha"]
        alpha_values = alpha_column.values

        alpha_sim_col = model.parameter_table.loc[:,"alpha_sim"]
        alpha_sim_vals = alpha_sim_col.values

    # Save estimated parameter values to file
    if params.continuous:
        param_path = os.path.join(file_path, f"con_n{n}_params.txt")
    else:
        param_path = os.path.join(file_path, f"bin_b{params.beta_val}_params.txt")

    with open(param_path, "w") as f:
        if not params.continuous:
            f.write(f"Mean est. beta: {np.mean(beta_values)} \nStd est. beta: {np.std(beta_values)} \n")
            f.write("\nBeta values:\n")
            f.write(f"\n{beta_values}\n \n")
            f.write(f"Mean est. log beta: {np.nanmean(log_beta_values)} \nStd est. beta: {np.nanstd(log_beta_values)} \n")
            f.write("\nLog beta values:\n")
            f.write(f"\n{log_beta_values}\n \n")

        f.write("Estimated alpha values:\n")
        f.write(f"\n{alpha_values}\n \n")
        f.write("Simulated alpha values:\n")
        f.write(f"\n{alpha_sim_vals}")
        if params.model_type == dual_lr_rw:
            f.write("\nEstimated alpha_n values:\n")
            f.write(f"\n{alpha_n_values}\n \n")
            f.write("Simulated alpha_n values:\n")
            f.write(f"\n{alpha_n_sim_vals}")

    # close file
    sys.stdout.close()
    sys.stdout = default_stdout

    # close all matplotlib plots
    plt.close('all')

    # increment
    run = run + 1

print("Finished")
