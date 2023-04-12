"""
    File name: main_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This is the code used to run BLT model simulations and fits
"""
import sys
sys.path.insert(0, 'DMpy/')

# # fix to deal with theano Exception on certain computers
# import theano
# theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, shapiro

from learning_BLT import dual_lr_rw

import params
from models_BLT import define_model, model_simulation, fit_model, plot_trajectories
from utils_BLT import get_model_stats

# create the model
model, l_values, o_values = define_model(model_type=params.model_type, continuous=params.continuous)

# run simulation
model_simulation(model, l_values, o_values, continuous=params.continuous, recover=True, sim_plot=True)

# fit model to real data
fit_model(model, continuous=params.continuous, plot=True)

# calculate likelihoods
get_model_stats(model, params.subID, params.n_outcomes, params.continuous, params.data_path, params.fit_method)

# plot trajectories for each participant
predictions = []
simulations = []

for s in params.subjects:
    p, s = plot_trajectories(s)
    predictions.append(p)
    simulations.append(s)

# # Get standard deviation of residuals and normality test
# residuals = np.array(predictions) - np.array(simulations)
# for n in range(len(params.subID)):
#     print(f"{n}: {np.std(residuals[n])}")
#     stat, p = shapiro(residuals[n])
#     print(f"p: {p}")
# print(f"Overall: {np.std(residuals)}")
# stat, p = shapiro(residuals)
# print(f"p: {p}")

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
