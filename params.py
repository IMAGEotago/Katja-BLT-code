"""
    File name: params.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains the parameters used by main_BLT.py.
                 Parameters defined in this file:
                 - continuous: determines whether model is continuous
                 - sim_path: filepath where results from simulation are written to
                 - mat_file: filepath to the matlab data file
                 - data_path: filepath to csv file containing data for model fitting
                 - outcomes: numpy array of trial outcomes
                 - n_outcomes: number of outcomes (also number of trials)
                 - n_subjects: number of subjects to be simulated
                 - sim_noise: standard deviation of gaussian noise to be used for simulation
                 - beta_val: beta value to be used for binary model
                 - fit_method: method to be used for model fitting
"""
import numpy as np
import os

from utils_BLT import get_BLT_data

# continuous stores boolean value determining whether model is continuous (True) or binary (False)
continuous = False

# sim_path stores the filepath where results from simulation are written to
sim_path = 'output_files/test_blt_responses.csv'

# mat_file stores the filepath to the .mat file containing the relevant data
# TODO: change to relevant filepath as needed
fileDir = os.path.dirname(os.path.realpath('__file__'))
mat_file = os.path.join(fileDir, '../../test_data/testKB_task_BLT_2021_03_09_130052.mat')
mat_file = os.path.abspath(os.path.realpath(mat_file))

# data_path stores the path to the csv file containing the data for model fitting
# outcomes stores the list of trial outcomes as an np.array
# TODO: modify filepath based on subject ID, manage multiple subject IDs?
data_path = 'output_files/subXX_data.csv'
outcomes = get_BLT_data(mat_file, data_path, 1, continuous)

# n_outcomes stores the number of outcomes
n_outcomes = len(outcomes)

# n_subjects stores the number of subjects to be simulated
n_subjects = 500

# standard deviation of gaussian noise to be used for simulation
sim_noise = 0.05

# beta value to be used for binary model
beta_val = 1

# fit_method stores the method to be used for model Fitting
fit_method = 'MLE'
