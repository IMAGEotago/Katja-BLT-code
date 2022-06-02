"""
    File name: pca.py
    Author: Katja Brand
    Created: 4/05/2022
    Description: Runs PCA on model values and questionnaire data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from scipy.stats import norm, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('input_files/pca_input.csv', header=0, index_col=0)

n_participants = 15

# Separate out features
features = ['STAI-T','STAI-S','GAD-7','ASI-3','CESD','PANAST-P','PANAST-N','FSS',
'CD_RISC-25','GSE','MAIA','PCS-Resp','PVAQ']

scores = df.loc[:,features].values

# Standardise scores
scores = StandardScaler().fit_transform(scores)

# Run PCA
pca = PCA()
principalComponents = pca.fit_transform(scores)
principalDf = pd.DataFrame(data = principalComponents)

# Get explained variance
exp_var = pca.explained_variance_ratio_

# Get coefficients/loadings
loadings = pd.DataFrame(pca.components_.T, columns=['PC1','PC2','PC3','PC4','PC5',
'PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'], index=features)
loadings = pd.DataFrame.abs(loadings)

# Display coefficients for principal components
x = np.arange(len(features))
plt.figure(figsize=(13,5))
plt.bar(x-0.2, loadings.loc[:,'PC1'], width=0.4, label='PC1')
plt.bar(x+0.2, loadings.loc[:,'PC2'], width=0.4, label='PC2')
plt.xticks(x, features)
plt.legend()
plt.show()

# Shuffle to create null distribution
n_shuffle = 1000 #number of times to shuffle
shuffled_scores = scores # create copy of scores to hold shuffled scores
pca_null_results = np.zeros((n_shuffle, n_participants,len(features)), dtype=np.ndarray) # array to hold PCA null results
pca_null_explained = np.zeros((n_shuffle, len(features)), dtype=np.ndarray) # array to hold explained variance for each PCA

for i in range(n_shuffle):
    # Shuffle each column
    for column in shuffled_scores:
        random.shuffle(column)
    # Run PCA on shuffled data
    pca_null = PCA()
    pca_null_results[i] = pca_null.fit_transform(shuffled_scores)
    pca_null_explained[i] = pca_null.explained_variance_ratio_

pca_null_explained = np.array(pca_null_explained, dtype=np.float64)

# Calculate mean and std of explained variance for each principal component across PCAs
ev_mean = np.mean(pca_null_explained, axis=0)
ev_std = np.std(pca_null_explained, axis=0)

# Calculate 95% confidence interval for the explained variance
ev_ci = norm.interval(alpha=0.95, loc=ev_mean, scale=ev_std)

# Calculate percentile for explained variance for each component
ev_perc = norm.cdf(exp_var, loc=ev_mean, scale=ev_std)

# Check if components are significant
for c in range(len(exp_var)):
    v = exp_var[c] # explained variance of the principal component
    ci_min = ev_ci[0][c] # minimum CI threshold for that component
    ci_max = ev_ci[1][c] # maximum CI threshold for that component
    if (v > ci_max):
        print(f"PC{c+1} is significant with explained variance of {np.round(v, 3)}")
        print(f"CI is {np.round(ev_ci[0][c], 3)}-{np.round(ev_ci[1][c], 3)}, p = {1-ev_perc[c]:.3e}")
    else:
        print(f"PC{c+1} is not significant")

# Correlate significant prinicipal components with model parameters
model_features = ['alpha-c', 'alpha-b', 'certainty']
model_scores = df.loc[:,model_features].values

pc_scores = principalDf.loc[:,0] #PC1 only

for i in range(len(model_features)):
    (r, p) = pearsonr(model_scores[:,i], pc_scores)
    print(f'\nCorrelation between PC1 and {model_features[i]}, r = {np.round(r,3)}    p = {p:.3e}')
