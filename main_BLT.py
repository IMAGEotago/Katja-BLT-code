"""
    File name: main_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This is the code used to run BLT model simulations and fits
"""
import sys
sys.path.insert(0, 'DMpy/')

# fix to deal with theano Exception on certain computers
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from learning_BLT import dual_lr_rw

import params
from models_BLT import define_model, model_simulation, fit_model, plot_trajectories
from utils_BLT import get_model_stats

# print type of model, number of subjects and trials
if params.continuous:
    print(f"Running continuous model with {params.n_subjects} subjects and {params.n_outcomes} trials")
else:
    print(f"Running binary model with {params.n_subjects} subjects and {params.n_outcomes} trials")

# create the model
model, l_values, o_values = define_model(model_type=params.model_type, continuous=params.continuous)

# run simulation
model_simulation(model, l_values, o_values, continuous=params.continuous, recover=True, sim_plot=True)

# # calculate likelihoods
# get_model_stats(model, params.n_subjects, params.n_outcomes, params.continuous)

# fit model to real data
fit_model(model, continuous=params.continuous, plot=True)

# calculate likelihoods
get_model_stats(model, len(params.subjects), params.n_outcomes, params.continuous)

# plot trajectories for each participant
predictions = []
simulations = []

for s in params.subjects:
    p, s = plot_trajectories(s)
    predictions.append(p)
    simulations.append(s)

# calculate and plot average trajectories
mean_prediction = np.mean(np.array(predictions), axis=0)
mean_simulation = np.mean(np.array(simulations), axis=0)

r_val, p_val = pearsonr(mean_prediction, mean_simulation)
print(f"R = {np.round(r_val, 6)}, p = {p_val}")

plt.figure(figsize=(15,4))
plt.plot(mean_prediction, '-', c='tab:red', label="Mean raw predictions")
plt.plot(np.arange(params.n_outcomes), mean_simulation, '-', c='black', label="Mean model predictions")
# if params.model_type == dual_lr_rw: #plots the outcomes as dark red circles
#     plt.plot(params.outcomes['Outcome'].values, 'o', c='darkred', alpha=0.8)
# else:
#     plt.plot(params.outcomes, 'o', c='darkred', alpha=0.8)
plt.text(75, 0.8, f'R = {np.round(r_val,4)}', fontsize='large')
plt.xlabel('Trial')
plt.legend()
plt.show()
