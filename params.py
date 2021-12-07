"""
    File name: params.py
    Author: Katja Brand
    Created: 4/06/2021
    Description: This contains the parameters used by main_BLT.py.
                 Parameters defined in this file:
                 - continuous: determines whether model is continuous
                 - model_type: the learning function to be used for the model
                 - subID: subject ID for data being used
                 - subjects: list of each subject to be used for model fitting
                 - sim_path: filepath to csv file where results from simulation are written to
                 - data_path: filepath to csv file containing data for model fitting
                 - outcomes: numpy array of trial outcomes
                 - n_outcomes: number of outcomes (also number of trials)
                 - n_subjects: number of subjects to be simulated
                 - sim_noise: standard deviation of gaussian noise to be used for simulation of continuous model
                 - beta_val: beta value to be used for binary model
                 - fit_method: method to be used for model fitting
"""
import numpy as np
import os
import pandas as pd

from learning_BLT import rescorla_wagner, dual_lr_rw
from utils_BLT import Subject, get_BLT_data

# continuous stores boolean value determining whether model is continuous (True) or binary (False)
continuous = False

# model_type stores the name of the learning function to be used for the model
model_type = dual_lr_rw

# subject ID
subID = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',
        '0011', '0012', '0013', '0014', '0015', '0016']

# sim_path stores the filepath where results from simulation are written to
sim_path = "output_files/sim_blt_responses.csv"

# get the filepath to the .mat file containing the data for each listed subject
fileDir = os.path.dirname(os.path.realpath('__file__'))
subjects = []

# for each subject, extract data from file and create a Subject object
for id in subID:
    if subID == 'test':
        mat_file = os.path.join(fileDir, '../../test_data/testKB_task_BLT_2021_03_09_130052.mat')
    else: #TODO: put in try/catch loop?
        mat_file = os.path.join(fileDir, f'../../../../OneDrive/data/sub-{id}/beh/sub-{id}_task-BLT_beh.mat')
    mat_file = os.path.abspath(os.path.realpath(mat_file))
    df, outcomes, resist = get_BLT_data(mat_file, id, continuous)
    subjects.append(Subject(id, df, outcomes, resist))

# data_path holds filepath to csv file containing data from all subjects for model fitting
data_path = "output_files/subject_data.csv"

# concatenate data from all subjects into a single dataframe (subject_data)
frames = []
for s in subjects:
    frames.append(s.df)

subject_data = pd.concat(frames)
subject_data.to_csv(data_path, index=False)

# n_outcomes stores the number of outcomes (note: will be based on outcomes from last subject)
n_outcomes = len(outcomes)

# convert outcomes to dataframe when using dual_lr_rw
if model_type == dual_lr_rw:
    outcomes = pd.DataFrame({'Outcome':outcomes, 'Resistance':resist})

# n_subjects stores the number of subjects to be simulated
n_subjects = 500

# standard deviation of gaussian noise to be used for simulation
sim_noise = 0.2

# beta value to be used for binary model
beta_val = 5

# fit_method stores the method to be used for model Fitting
fit_method = 'MAP'
