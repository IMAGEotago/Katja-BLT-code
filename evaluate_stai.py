"""
    File name: evaluate_stai.py
    Author: Katja Brand
    Created: 14/01/2022
    Description: Loads data from excel spreadsheet and evaluates it based on the
                 STAI-T or STAI-S scoring criteria.
"""

import numpy as np
import os
import pandas as pd

# type of STAI - either T or S
type = "S"

# weightings for the STAI-T (0 = keep, 1 = flip)
t_weights = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0])

# weightings for the STAI-S
s_weights = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])

def evaluate_stai(responses, weights):
    """
        Evaluates responses to the STAI and returns the score.
        Arguments:
            - responses: participant's responses in form of pandas series
            - weights: weighting for each question, used for scoring
        Returns:
            - score: participant's score
    """
    responses = responses.to_numpy()

    for i in range(len(responses)):
        if weights[i] == 1:
            responses[i] = 5 - responses[i] #flip responses when weights is 1

    score = np.sum(responses)
    return score

# get filepath to excel sheet
fileDir = os.path.dirname(os.path.realpath('__file__'))

if type == "T":
    print("Running STAI-T evaluation:")
    csv_path = os.path.join(fileDir, 'input_files\\anxiety_scores_t.csv')
elif type == "S":
    print("Running STAI-S evaluation:")
    csv_path = os.path.join(fileDir, 'input_files\\anxiety_scores_s.csv')
else:
    print("Error: wrong type")

stai_df = pd.read_csv(csv_path, header=0, index_col=0)

for index, row in stai_df.iterrows():
    if type == "T":
        score = evaluate_stai(row, t_weights)
    elif type == "S":
        score = evaluate_stai(row, s_weights)
    else:
        print("Error: wrong type")

    print(f"Participant {index} returned score {score} on STAI-{type}")
