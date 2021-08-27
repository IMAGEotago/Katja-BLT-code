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
from models_BLT import define_model, model_simulation

# save all output to text file
fileDir = os.path.dirname(os.path.realpath('__file__'))
file_path = ""
if params.continuous:
    file_path = os.path.join(fileDir, "../sim_results/continuous")
    text_path = os.path.join(file_path, f"con_n{params.sim_noise*100}_log.txt")
else:
    file_path = os.path.join(fileDir, "../sim_results/binary")
    text_path = os.path.join(file_path, f"bin_b{params.beta_val}_log.txt")

default_stdout = sys.stdout
sys.stdout = open(text_path, "w")

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
model_simulation(model, values, continuous=params.continuous, recover=True, sim_plot=True)

#msave plots
figs = [plt.figure(n) for n in plt.get_fignums()]
n = 1
for fig in figs:
    if params.continuous:
        fig_name = f"con_n{params.sim_noise*100}_f{n}"
    else:
        fig_name = f"bin_b{params.beta_val}_f{n}"

    fig_path = os.path.join(file_path, f"{fig_name}.png")
    fig.savefig(fig_path)
    n = n+1

# print estimated beta mean + sd
beta_column = model.parameter_table.loc[:,"beta"]
beta_values = beta_column.values
print(f"\n \nMean est. beta: {np.mean(beta_values)}")
print(f"Std est. beta: {np.std(beta_values)}")

alpha_column = model.parameter_table.loc[:,"alpha"]
alpha_values = alpha_column.values

print("\n \nBeta values:")
print(beta_values)
print("\n \nAlpha values:")
print(alpha_values)
# Save estimated parameter values to file
# TODO: only saves first line
# if params.continuous:
#     param_path = os.path.join(file_path, f"con_n{params.sim_noise*100}_params.txt")
# else:
#     param_path = os.path.join(file_path, f"bin_b{params.beta_val}_params.txt")
#
# with open(param_path, "w") as f:
#     f.write(f"Mean est. beta: {np.mean(beta_values)}            Std est. beta: {np.std(beta_values)} \n")
#     f.write(beta_values)
#
#     f.write("\n \n Alpha values:")
#     f.write(alpha_values)

# close file
sys.stdout.close()
sys.stdout = default_stdout
