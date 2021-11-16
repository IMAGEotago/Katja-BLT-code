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
from utils_BLT import likelihood_ratio

# print type of model, number of subjects and trials
if params.continuous:
    print(f"Running continuous model with {params.n_subjects} subjects and {params.n_outcomes} trials")
else:
    print(f"Running binary model with {params.n_subjects} subjects and {params.n_outcomes} trials")

# create the model
model, l_values, o_values = define_model(model_type=params.model_type, continuous=params.continuous)

# run simulation
model_simulation(model, l_values, o_values, continuous=params.continuous, recover=True, sim_plot=True)

# calculate likelihoods
# TODO: this part takes a long time with many subjects, put into separate function, write output to file
individual_fits = model.individual_fits()

for s in range(params.n_subjects):
    subject = s + 1
    log_likelihood = individual_fits['logp'][(subject*params.n_outcomes) - 1]

    # likelihood ratio test
    lr, p = likelihood_ratio(80*np.log(0.5), log_likelihood)
    print(f"\nSubject {subject}")
    print(f"Model log likelihood: {log_likelihood}")
    print(f"lr: {lr}")
    print("p: %.30f" %p)
    print(f"pseudo-r2 = {1 - (log_likelihood / (80*np.log(0.5)))}")

# fit model to real data
fit_model(model, continuous=params.continuous, plot=True)

#plot trajectories
for s in params.subjects:
    plot_trajectories(s)
