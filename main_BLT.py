"""
    File name: main_BLT.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This is the code used to run BLT model simulations and fits
"""
import sys
sys.path.insert(0, 'DMpy/')

import params
from functions_BLT import define_model

print(params.mat_file)
print(params.outcomes)
print(params.n_outcomes)

model = define_model(continuous=False)

# TODO: simulate and recover from model
