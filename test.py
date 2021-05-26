import sys
sys.path.insert(0, 'DMpy/')
#print(sys.path)

# IPython reloading if modifying module code
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from DMpy import DMModel, Parameter
from DMpy.learning import dual_lr_qlearning
from DMpy.observation import softmax
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('white')
import os

# Load the data
outcomes = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1.,
                     0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                     1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.,
                     1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
                     0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1.,
                     0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1.,
                     1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1.,
                     0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 1.])
n_outcomes = len(outcomes)
#outcomes = pd.DataFrame(outcomes,columns=['Outcome'])
#outcomes["Subject"] = "s01"
#outcomes.loc[100:, "Subject"] = "s02"
#outcomes["Run"] = "r01"
#outcomes.loc[100:, "Run"] = "r02"
#outcomes = np.tile(outcomes, (100, 1))

# Set parameters
value = Parameter('value', 'fixed', mean=0.5, dynamic=True)
alpha_p = Parameter('alpha_p', 'fixed', mean=0.3)
alpha_n = Parameter('alpha_n', 'normal', lower_bound=0, upper_bound=1, mean=0.5,variance=0.1)
beta = Parameter('beta', 'fixed', mean=3)

# Create model
model_dual_lr = DMModel(dual_lr_qlearning, [value, alpha_p, alpha_n], softmax, [beta])

# Define parameter values for simulation

# Number of participants in each group
n_groupA = 50
n_groupB = 50

alpha_p_values = np.random.normal(0.3, 0.05, n_groupA + n_groupB)

alpha_n_values = np.concatenate([np.random.normal(0.3, 0.05, n_groupA), np.random.normal(0.7, 0.05, n_groupB)])

# Plot parameter distributions
sns.histplot(alpha_p_values, label='alpha_p', kde=True)
sns.histplot(alpha_n_values, label='alpha_n', kde=True, color='green')
plt.legend()
plt.xlabel("Parameter value")
plt.ylabel("Density")
#plt.tight_layout()
plt.show()

# Simulate data
_, sim_dual_lr = model_dual_lr.simulate(outcomes=outcomes,
                                        learning_parameters={
                                            'value': [0.5] * int(n_groupA + n_groupB),
                                            'alpha_p': alpha_p_values,
                                            'alpha_n': alpha_n_values},
                                        observation_parameters={
                                            'beta': [3] * int(n_groupA + n_groupB)},
                                        output_file='output_files/example_responses.txt')

#display trace of first subject from each group
low_alpha_results = model_dual_lr.simulation_results['value'][0:n_outcomes]
high_alpha_results = model_dual_lr.simulation_results['value'][n_groupA*n_outcomes:(n_groupA + 1)*n_outcomes]

x = np.arange(n_outcomes)

# Plot example from simulation
plt.figure(figsize=(15, 3))
#plt.plot(model_dual_lr.simulated['sim_results']['value'][:, 0], label='Low alpha_n')
plt.plot(x, low_alpha_results, label='Low alpha_n')
#plt.plot(model_dual_lr.simulated['sim_results']['value'][:, n_groupA], label='High alpha_n')
plt.plot(x, high_alpha_results, label='High alpha_n')
plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
plt.legend()
plt.xlabel('Trial')
plt.ylabel('Estimated value')
#plt.tight_layout()
plt.show()

# Fit the simulated data
#model_dual_lr.fit(sim_dual_lr, fit_method='variational', fit_kwargs=dict(n=30000),
#                    logp_method='ll', hierarchical=True, recovery=True)
model_dual_lr.fit(sim_dual_lr, fit_method='variational', fit_kwargs=dict(n=30000),
                hierarchical=True, recovery=True)
