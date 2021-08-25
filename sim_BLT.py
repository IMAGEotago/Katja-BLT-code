"""
    File name: sim_BLT.py
    Author: Katja Brand
    Created: 25/08/2021
    Description: This is a script for running multiple model simulations and
                 saving the results.
"""
import sys
sys.path.insert(0, 'DMpy/')

import matplotlib.pyplot as plt
import os

import params
from models_BLT import define_model, model_simulation

# print relevant model parameters
if params.continuous:
    print(f"Running continuous model with {params.n_subjects} subjects and {params.n_outcomes} trials")
    print(f"Noise SD: {params.sim_noise}")
else:
    print(f"Running binary model with {params.n_subjects} subjects and {params.n_outcomes} trials")
    print(f"Beta: {params.beta_val}")

# create model
model, values = define_model(continuous=params.continuous)

# run simulation and parameter recovery
#TODO: iterate and save plots to folders
model_simulation(model, values, continuous=params.continuous, recover=True, sim_plot=True)

#save plots
#TODO: only saves first plot so far
fileDir = os.path.dirname(os.path.realpath('__file__'))

figs = [plt.figure(n) for n in plt.get_fignums()]
n = 1
for fig in figs:
    if params.continuous:
        fig_name = f"con_n{params.sim_noise*100}_f{n}"
        fig_path = os.path.join(fileDir, f"../sim_results/continuous/{fig_name}.png")
    else:
        fig_name = f"bin_b{params.beta_val}_f{n}"
        fig_path = os.path.join(fileDir, f"../sim_results/binary/{fig_name}.png")

    fig.savefig(fig_path)

#TODO: print estimated beta values/mean + sd
