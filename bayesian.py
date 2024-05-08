#%% imports
import sys; sys.path.insert(0, '..')
import PEUQSE
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
import os
import PEUQSE.UserInput
from multiprocessing import Pool
import shutil

# test importing variable filename

import UserSettings
import gpmodelFile
#%%
# this import line imports 2 functions into the PEUQSE UserInput
# importlib.__import__(f'{UserSettings.surrogate_model_filename}')

#%% functions
# move functions to model file, name it specific for GP 

def parameter_range_perturbation(inputs):
    param_name, param_value, num_count = inputs
    if param_value < 0:
        UserSettings.low_vals[num_count] = UserSettings.low_vals[num_count] * (1+.1)
        UserSettings.up_vals[num_count] = UserSettings.up_vals[num_count] * (1-.1)
    else:
        UserSettings.low_vals[num_count] = UserSettings.low_vals[num_count] * (1-.1)
        UserSettings.up_vals[num_count] = UserSettings.up_vals[num_count] * (1+.1)
    PEUQSE.UserInput.model['InputParameterPriorValues_upperBounds'] = UserSettings.up_vals # for advanced users to change
    PEUQSE.UserInput.model['InputParameterPriorValues_lowerBounds'] = UserSettings.low_vals # for advanced users to change
    path = os.getcwd()
    os.chdir(path + '/PE/' + str(param_name))

    bayesian_sim()


def create_geweke_plots():
    pass

def bayesian_sim():
    # %% change directory
    path = os.getcwd()
    os.chdir(path + '/PE')

    path = os.getcwd()
    new_dir = f'{UserSettings.sampling_method}'
    try:
        os.mkdir(new_dir)
    except OSError:
        print(' ')
    os.chdir(path + f'/{new_dir}')
    

    # PEUQSE.UserInput.model['simulatedResponses_upperBounds'] = up_prop
    # PEUQSE.UserInput.model['simulatedResponses_lowerBounds'] = low_prop



    PE_object = PEUQSE.parameter_estimation(PEUQSE.UserInput) 
    date = time.strftime("%Y%m%d", time.localtime())
    start_time = time.monotonic()
    shutil.copy('../../UserSettings.py', f'UserSettings_{date}.py')


    if UserSettings.sampling_method == "ESS":
        print('Starting Ensemble Slice Sampling...')
        mcmc_output = PE_object.doEnsembleSliceSampling(walkerInitialDistributionSpread=UserSettings.ESSwalkerInitialDistributionSpread)

    if UserSettings.sampling_method == "MH":
        print('Starting Metropolis Hastings Sampling...')
        mcmc_output = PE_object.doMetropolisHastings()
    
    if UserSettings.sampling_method == "EJS":
        print('Starting Ensemble Jump Sampling...')
        mcmc_output = PE_object.doEnsembleJumpSampling()

    if UserSettings.sampling_method == "astroidal":
        print('Starting MultiStart Astroidal Sampling...')
        mcmc_output = PE_object.doMultiStart()
    
    if UserSettings.sampling_method == "sobol":
        print('Starting MultiStart Sobol Sampling...')
        mcmc_output = PE_object.doMultiStart()

    if UserSettings.sampling_method == "shell":
        print('Starting MultiStart Shell Sampling...')
        mcmc_output = PE_object.doMultiStart()

    if UserSettings.sampling_method == "uniform":
        print('Starting MultiStart Uniform Sampling...')
        mcmc_output = PE_object.doMultiStart()
    
    if UserSettings.sampling_method == "gridsearch":
        print('Starting MultiStart grid Sampling...')
        mcmc_output = PE_object.doMultiStart()
    
    # for gridsearch multistart (errors) not good
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True

    run_time = time.monotonic() - start_time 
    str_run_time = str(datetime.timedelta(seconds=run_time))
    # print (f'Run time: {str_run_time}') to a text file
    with open(f'{UserSettings.sampling_method}_run_time.txt', 'w') as f:
        f.write(f'Run time (sec): {str_run_time}')
    print(f'Run time: {str_run_time}')
    #%%
    PEUQSE.dillpickleAnObject(PE_object, f'BPE_object_{date}')
    PE_object.createAllPlots()
    # try:
    #     mcmc_post_burnin = PE_object.post_burn_in_samples
    #     fig, axes = plt.subplots(len(UserSettings.feature_headers), 1, tight_layout=True, squeeze=True, figsize=(10, 25))
    #     for num, feature in enumerate(UserSettings.feature_headers):
    #         z = pm.geweke(mcmc_post_burnin[:, num], intervals=35)
    #         plt.subplot(len(UserSettings.feature_headers), 1, num+1)
    #         plt.gca().set_title(feature)
    #         plt.scatter(*(z).T)
    #         plt.hlines([-1,1], 0, round(z[-1][0]), linestyles='dotted')
    #         plt.xlim(0, round(z[-1][0]))
    #     plt.gcf().savefig(f'Geweke.png')
    # except:
    #     print('Failed when creating gewekes indices plots')

    os.chdir('..')
    del PE_object




#%% main
if __name__ == '__main__':
    folders = ['PE'] # TODO: Add to this as I go
    for folder in folders:
        try:
            os.mkdir(folder)
        except OSError as error:
            print(' ')
    path = os.getcwd()
    os.chdir(path + '/PE')

    # for folder in UserSettings.feature_headers:
    #     try:
    #         os.mkdir(folder)
    #     except OSError as error:
    #         print(' ')
    os.chdir('..')

    # pool = Pool()
    # pool.map(bayesian_sim, )
    bayesian_sim()
    # inputs = [(x,y,z) for x,y,z in zip(UserSettings.feature_headers, UserSettings.best_param_set, range(len(UserSettings.feature_headers)))]
    # pool.map(parameter_range_perturbation, inputs)


# run code
# label_headers = ['cohesive', 'poisson ratio', 'C12', 'C11',
#                         'bulk_modulus', 'C44', 'surface100', 'density298.15']

# param_values = np.array([15.069252,	22.790225,	3.932003,	0.257858,	2.775952, 1.354304,
#             1.194269,	-2.063201, 0.708031, -16.824827, 1.330723,	0.927891, 7.719486, 1.570365])

# type_sobol = ['0.25%', '0.5%', '1%', '2%', '4%']



    

# %%
