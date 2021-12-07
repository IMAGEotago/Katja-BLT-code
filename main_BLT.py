"""
    File name: main_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This is the code used to run BLT model simulations and fits
"""
import sys
sys.path.insert(0, 'DMpy/')

# fix to deal with theano Exception on certain computers
# import theano
# theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

import numpy as np
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

# #plot trajectories
# for s in params.subjects:
#     plot_trajectories(s)
