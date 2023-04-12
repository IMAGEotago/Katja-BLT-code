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
from utils_BLT import Subject, get_BLT_data, get_certainty, get_proportion_correct

# continuous stores boolean value determining whether model is continuous (True) or binary (False)
continuous = True

# model_type stores the name of the learning function to be used for the model
model_type = rescorla_wagner

# subject ID
subID = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0009', '0010',
        '0011', '0012', '0013', '0014', '0015', '0016']
# subID = 'pilot'
# subID = ['test','test2'] # Use for testing

# sim_path stores the filepath where results from simulation are written to
sim_path = "output_files/sim_blt_responses.csv"

# get the filepath to the .mat file containing the data for each listed subject
fileDir = os.path.dirname(os.path.realpath('__file__'))
subjects = []

# for each subject, extract data from file and create a Subject object
if subID == 'pilot':
    xls_file = os.path.join(fileDir, '../../code/data/pilot/PBIHB_pilots_beh.xlsx')
    xls_file = os.path.abspath(os.path.realpath(xls_file))
    pilot_responses = pd.read_excel(xls_file, sheet_name='y', header=None, index_col=None)
    pilot_outcomes = pd.read_excel(xls_file, sheet_name='u', header=None, index_col=None)
    print(pilot_responses)
    print(pilot_outcomes)
    for p in range(8):
        outcomes = pilot_outcomes.iloc[p]
        responses = pilot_responses.iloc[p]
        df = pd.DataFrame({'Outcome':outcomes, 'Response':responses, 'Subject':p})
        print(f"pilot {p} df: \n{df}")
        subjects.append(Subject(p, df, outcomes, None))
else:
    for id in subID:
        if 'test' in id:
            mat_file = os.path.join(fileDir, 'input_files/test_data.mat')
        else:
            mat_file = os.path.join(fileDir, f'../../code/data/sub-{id}/beh/sub-{id}_task-BLT_beh.mat')
        mat_file = os.path.abspath(os.path.realpath(mat_file))
        df, outcomes, resist = get_BLT_data(mat_file, id, continuous)
        subjects.append(Subject(id, df, outcomes, resist))
        get_certainty(id, df) #print average certainty for each subject

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

# # get proportion of correct responses for each trial
# prop_df = get_proportion_correct(subjects, n_outcomes, True)
# prop_df.to_csv("output_files/proportions.csv", index=False)

# convert outcomes to dataframe when using dual_lr_rw
if model_type == dual_lr_rw:
    outcomes = pd.DataFrame({'Outcome':outcomes, 'Resistance':resist})

# n_subjects stores the number of subjects to be simulated
n_subjects = 500

# standard deviation of gaussian noise to be used for simulation
sim_noise = 0.01

# beta value to be used for binary model simulations
beta_val = 5

# fit_method stores the method to be used for model fitting e.g. 'MLE', 'MAP'
fit_method = 'MAP'
