# Katja-BLT-code

## About
This code was written to simulate and interpret data from the Breathing Learning Task (BLT).

## How to use
### Install repository
Create a local version of the repository using the commands:
```
git clone https://github.com/IMAGEotago/Katja-BLT-code.git
cd Katja-BLT-code/
git submodule update --init
```

### Get data
Within the 'input_files' folder put the 'data' folder containing the subject data.

### Create virtual environment
Create a conda environment using the requirements.yml file. The command for this is:
```
conda env create -f requirements.yml
conda activate blt_env
```
This will create a virtual environment called 'blt_env' that contains the python packages necessary to run the code.

### Run the program
Open and run the script 'main_BLT.py' in the 'blt_env' virtual environment. This script will run model simulation and parameter recovery followed by model fitting for each subject provided. To skip the simulation and parameter recovery steps, comment out line 28 in the script.

The 'params.py' file contains parameters for model simulation and fitting. Some useful parameters that can be changed include:
- `continuous`: this can be set to `True` to run the continous model and `False` to run the binary model
- `model_type`: this can be set to `rescorla_wagner` (single learning rate Rescorla-Wagner model) or `dual_lr_rw` (Dual learning rate Rescorla-Wagner model)
- `subID`: this takes an array of strings containing the subject IDs to use in model fitting
- `n_subjects`: this sets the number of subjects to be used in model simulations
- `sim_noise`: this sets the standard deviation of gaussian noise to be used in continuous model simulations
- `beta_val`: this sets the beta parameter to be used in binary model simulations

### Note on installing
When creating the virtual environment on a Macbook, there may be an issue with some files relating to the netCDF4 package (which is installed with the PyMC3 package) not being installed properly. This causes an ImportError when the 'main_BLT.py' script is run. The current fix for this is to uninstall and reinstall the PyMC3 package after the environment has been created using the requirements.yml file as described above. This can be done using the commands:
```
conda uninstall pymc3
conda install pymc3
```
