import sys
sys.path.insert(0, 'DMpy/')

import numpy as np
from DMpy import DMModel, Parameter
from DMpy.learning import rescorla_wagner
from DMpy.observation import softmax
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# import data

# open matlab file
fileDir = os.path.dirname(os.path.realpath('__file__'))
mat_file = os.path.join(fileDir, '../test_data/testKB_task_BLT_2021_03_09_130052.mat')
mat_file = os.path.abspath(os.path.realpath(mat_file))

mat_contents = sio.loadmat(mat_file) #reads matlab variables into python dict

params = mat_contents['params'][0][0]

outcomes = np.array(params[10]).reshape(len(params[10])) #check right one, could be at 7
print(outcomes)

# outcomes = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                      0., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
#                      0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1.,
#                      0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
#                      1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1.,
#                      1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0.,
#                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.,
#                      0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1.,
#                      0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
#                      0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1.,
#                      1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1.,
#                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
#                      0., 0., 0., 0., 1.])

n_outcomes = len(outcomes)


# set parameters
value = Parameter('value', 'fixed', mean=0.5, dynamic=True)
alpha = Parameter('alpha', 'uniform', lower_bound=0.0, upper_bound=1.0)
beta = Parameter('beta', 'flat', mean=1) #TODO: change to appropriate value

n_subjects = 25
value_values = [0.5] * n_subjects
alpha_values = np.random.uniform(0.0, 1.0, n_subjects) #TODO: determine appropriate no. of samples
beta_values = [1] * n_subjects #TODO: determine appropriate value/s

# create model
# learning model is rescorla_wagner model with parameter alpha
# observation model is softmax with parameter beta
model = DMModel(rescorla_wagner,[value, alpha], softmax, [beta])

# simulate data for parameter recovery
_, sim_rw = model.simulate(outcomes=outcomes,
                           n_subjects=n_subjects,
                           output_file='test_blt_responses.csv',
                           learning_parameters={'value' : value_values,
                                                'alpha' : alpha_values},
                           observation_parameters={'beta' : beta_values})

# plot simulated data
x = np.arange(n_outcomes)

a1 = model.simulation_results['alpha_sim'][0]
a2 = model.simulation_results['alpha_sim'][n_outcomes]
a3 = model.simulation_results['alpha_sim'][2*n_outcomes]

plt.figure(figsize=(15, 3))
plt.plot(x, model.simulation_results['value'][0:n_outcomes], label='alpha_s1 = %.3f' %a1)
plt.plot(x, model.simulation_results['value'][n_outcomes:n_outcomes*2], label='alpha_s2 = %.3f' %a2)
plt.plot(x, model.simulation_results['value'][n_outcomes*2:n_outcomes*3], label='alpha_s3 = %.3f' %a3)
plt.scatter(range(0, len(outcomes)), outcomes, facecolors='none', linewidths=1, color='black', alpha=0.5)
plt.legend()
plt.xlabel('Trial')
plt.ylabel('Estimated value')
plt.show()

# fit the simulated data and perform parameter recovery
model.fit(sim_rw, fit_method='MLE', fit_stats=True, recovery=True) #TODO: value for fit_kwargs?
