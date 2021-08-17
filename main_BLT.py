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

import params
from models_BLT import define_model, model_simulation, fit_model

# print type of model, number of subjects and trials
if params.continuous:
    print(f"Running continuous model with {params.n_subjects} subjects and {params.n_outcomes} trials")
else:
    print(f"Running binary model with {params.n_subjects} subjects and {params.n_outcomes} trials")

model, values = define_model(continuous=params.continuous)

model_simulation(model, values, continuous=params.continuous, recover=True, sim_plot=True)

# call function for fitting model to real data
#fit_model(model, continuous=params.continuous, plot=True)
