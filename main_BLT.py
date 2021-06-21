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

import params
from functions_BLT import define_model, model_simulation

# print(params.mat_file)
print(params.continuous)
print(params.outcomes)
print(params.n_outcomes)

model, values = define_model(continuous=params.continuous)

model_simulation(model, values, continuous=params.continuous, recover=True, sim_plot=True)
