# User Settings dependant on MD simulations
import sys; sys.path.insert(0, '..')
import numpy as np
import PEUQSE.UserInput
import pandas as pd

### VARIABLE KEY ###
# the number after the variable indicates what type of user should change the respective variable
# 0 - always changed
# 1 - commonly changed
# 2 - for advanced users with specific intentions

### main settings ###
# MD settings
metal_name = "au" # 0
forcefield_type = "aa" # 1; either course grain (cg) or all atom (aa)
# all labels = ["cohesive","poisson ratio","C12","C11","bulk_modulus","C44","surface100","surface110","surface111","VFE","density298.15"]
# label_headers = ["cohesive","poisson ratio","C12","C11","bulk_modulus","C44","surface100","surface110","surface111","VFE","density298.15"] # 0
label_headers = ["cohesive","poisson ratio","C12","C11","bulk_modulus","surface100","surface110","surface111","density298.15"]
feature_headers = ["rhoe", "alpha", "beta", "A", "B", "lamda", "kappa",
                   "Fn3", "F2", "Fe", "etha", "rhom", "lattice_constant", "fe"]
feature_err_headers = ["err_poisson_ratio", "err_surface100", "err_surface110", "err_surface111"] # 0 , adding error terms in the parameters TODO: Change the names to properties and parameters
features_math_expression = {'rhoe':r'$\rho_{e}$', 'alpha':r'$\alpha$', 
        'beta':r'$\beta$', 'A':'A', 'B':'B', 'lambda':r'$\lambda$', 'kappa':r'$\kappa$', 'Fn3':r'$Fn_{3}$', 
        'F2':r'$F_{2}$', 'F_e':r'$F_{e}$', 'etha':r'$\eta$', 'rhom':r'$\rho_{m}$', 'var13':'var13', 'fe':r'$f_{e}$'} # 1
features_err_math_expression = {"err_poisson_ratio":"e_PR", "err_surface100":"e_ST100", "err_surface110":'e_ST110', "err_surface111":"e_ST111"} 

datafile_name = "AA_Au_UQ - new_set.csv" # 0 
# Optimization result settings
best_param_set = np.array([12.40053,9.319661,4.266695,0.22,0.567516,1.01605,0.562943,-2.062738,1.693363,-1.419297,1.394784,0.747859,4.052219,1.260406]) #0 TODO: add the error mean values
best_prop_set = np.array([-3.853219513, 0.4636205204, 162.9177135 ,188.4854413 ,171.4402894 ,43.3336906 ,1377.126654 ,1524.09548 ,1451.439678 ,1.022390002 ,19.018803])
best_param_err_set = np.array([0.03898, -33.5897, -93.5897, 10.3846])
label_uncertainty_percents = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.07, 0.07, 0.02, 0.02]) # 0
target_values = np.array([-3.81,0.42,163,192,173,1540,1600,1480,19.3]) # 0 [-3.81,0.42,163,192,173,42.3,1540,1600,1480,1.02239,19.3]
label_uncertainty = np.array([0.057,0.0034,5.41,7.877,5.53,60.1,60.1,60.1,0.059]) # [0.057,0.0034,5.41,7.877,5.53,1.468,60.1,60.1,60.1,0.07,0.059]
# Surrogate Model settings
surrogate_model_filename = "contgpmodelFile.py" # 2
training_data_percent = 0.1 # TODO implement
num_points_training = 1000 # 2; 
gp_kernels = ['RBF', 'Exponential', 'Mat32', 'Mat52', 'RBF+RBF'] #2; options: 'RBF', 'Exponential', 'Mat32', 'Mat52', 'RBF+RBF', 'Cosine'
# Cross Validation settings
cross_validation_type = 'k3' # 1, this will determine what type of CV you will do, LOO will guide k=n-1 folds where k-folds can be specified by k followed by the number of folds, ex ['LOO', 'k3', 'k4', 'k5']
# Bayesian model settings
percent_bound_extension = 0 # 1; default 0 for no extension of bounds, 0.1 = 10% 
param_prior_type = "uniform" # 2
sampling_method = "ESS" # 0; choices are: ESS, MH, EJS, uniform, sobol, astroidal, shell, gridsearch
num_sample_points = 2000000 # 0;  number of samples can depend on the sampler
err_flag = False # 0; determines if correction values are used or not (True or False)
error_to_response_hash = {"err_poisson_ratio":"poisson ratio", "err_surface100":"surface100", "err_surface110":"surface110", "err_surface111":"surface111"} # 1; helps the gp model add in error uncertainties



### advanced user settings ###
#  for bayesian analysis, all lines are 2 difficulty
df = pd.read_csv(datafile_name, header=0) # change to call df only from UserSettings
# target_values = target_values[np.newaxis].T

if param_prior_type == 'uniform':
    param_uncertain = [-1] * len(feature_headers) # 2; uniform priors TODO:check this and automate this (Probably concatenate two lists where they are defined seperately)
param_err_uncertain = [0.003953, 84.9888, 84.9888, 74.4517] # 1; use normal priors for error terms, 1 std from the mean err 


low_vals = df[feature_headers].min().to_numpy() # currently upper and lower bounds are set to the bounds of the sobol data
up_vals = df[feature_headers].max().to_numpy()
low_err_vals = np.array([mu - 2*sig for (mu, sig) in zip(best_param_err_set, param_err_uncertain)])
up_err_vals = np.array([mu + 2*sig for (mu, sig) in zip(best_param_err_set, param_err_uncertain)])
low_vals = low_vals* [1+percent_bound_extension if x<0 else 1-percent_bound_extension for x in low_vals]
up_vals = up_vals* [1+percent_bound_extension if x>0 else 1-percent_bound_extension for x in up_vals]
param_std = list(df[feature_headers].std())

# TODO: Make the error correction terms final value a new variable so non-error files do not have trouble
# add error correction terms
if err_flag:
    feature_headers += feature_err_headers
    best_param_set = np.concatenate((best_param_set, best_param_err_set), axis=None)
    param_uncertain += param_err_uncertain
    features_math_expression.update(features_err_math_expression)
    low_vals = np.concatenate((low_vals,low_err_vals), axis=None)
    up_vals = np.concatenate((up_vals, up_err_vals), axis=None)


# PEUQSE section
PEUQSE.UserInput.responses['responses_observed'] = np.array([[x] for x in target_values])
PEUQSE.UserInput.responses['responses_observed_uncertainties'] = np.array([[x] for x in label_uncertainty]) #_percents

PEUQSE.UserInput.model['parameterNamesAndMathTypeExpressionsDict'] = features_math_expression
PEUQSE.UserInput.model['InputParameterPriorValues'] = best_param_set
PEUQSE.UserInput.model['InputParametersPriorValuesUncertainties'] = param_uncertain # 2; if not using uniform, specify the stds of each parameter for Gaussian Prior Distribution
PEUQSE.UserInput.model['InputParameterPriorValues_upperBounds'] = up_vals # for advanced users to change
PEUQSE.UserInput.model['InputParameterPriorValues_lowerBounds'] = low_vals # for advanced users to change
# (next 2) functions are created in gpmodelFile.py, change the model file if using differsent surrogate models than GP
PEUQSE.UserInput.model['simulateByInputParametersOnlyFunction'] = 'place holder' 
PEUQSE.UserInput.model['responses_simulation_uncertainties'] = 'place holder'
PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_samples'] = True
# mcmcm settings
if sampling_method == 'ESS':
    ESSwalkerInitialDistributionSpread = 0.25 # 2, the default is 1, lower values can be used if starting from the highest probability
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_parallel_sampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_continueSampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_length'] = num_sample_points
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_mode'] = 'unbiased'
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_burn_in'] = 'auto' # 2; if 'auto' is used, the burn in is set to 10% of the number of samples
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_random_seed'] = None # 1, there is a standard random seed used, use this to change from the standard seed, change to None to have no seed
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_coefficient'] = 1.5 # 2, 'auto' is the default of 2.0, 
    # PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_benchmark'] = 'MAP' # 2, can be 'auto' for outlier filtering, 'MAP' filtering for threshold filter starting at maximum probability, 'mu_AP' filtering for threshold filter starting at the mean probability

if sampling_method == 'MH':
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_parallel_sampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_continueSampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_length'] = num_sample_points
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_mode'] = 'unbiased'
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_burn_in'] = 'auto' # 2; if 'auto' is used, the burn in is set to 10% of the number of samples
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_random_seed'] = None # 1, there is a standard random seed used, use this to change from the standard seed, change to None to have no seed
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_coefficient'] = 1.5 # 2, 'auto' is the default of 2.0, 
    # PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_benchmark'] = 'MAP' # 2, can be 'auto' for outlier filtering, 'MAP' filtering for threshold filter starting at maximum probability, 'mu_AP' filtering for threshold filter starting at the mean probability

if sampling_method == 'EJS':
    EJSwalkerInitialDistributionSpread = 1 # 2, the default is 1, lower values can be used if starting from the highest probability
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_parallel_sampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_continueSampling'] = False
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_length'] = num_sample_points
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_mode'] = 'unbiased'
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_burn_in'] = 'auto' # 2; if 'auto' is used, the burn in is set to 10% of the number of samples
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_random_seed'] = None # 1, there is a standard random seed used, use this to change from the standard seed, change to None to have no seed
    PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_coefficient'] = 1.5 # 2, 'auto' is the default of 2.0, 
    # PEUQSE.UserInput.parameter_estimation_settings['mcmc_threshold_filter_benchmark'] = 'MAP' # 2, can be 'auto' for outlier filtering, 'MAP' filtering for threshold filter starting at maximum probability, 'mu_AP' filtering for threshold filter starting at the mean probability

if sampling_method == "uniform":
    PEUQSE.UserInput.parameter_estimation_settings['multistart_initialPointsDistributionType'] = 'uniform'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_searchType'] = 'getLogP' # 2; possible options for uniform optimization: getLogP , doOptimizeLogP , doOptimizeSSR , doOptimizeNegLogP
    PEUQSE.UserInput.parameter_estimation_settings['multistart_numStartPoints'] = num_sample_points
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearch_threshold_filter_coefficient'] = 2.0 #The lower this is, the more the points become filtered. It is not recommended to go below 2.0.
    PEUQSE.UserInput.parameter_estimation_settings['multistart_relativeInitialDistributionSpread'] = 0.2 # the lower this is, the more concentrated it is
    PEUQSE.UserInput.parameter_estimation_settings['scaling_uncertainties_type'] = "off"
    PEUQSE.UserInput.parameter_estimation_settings['multistart_checkPointFrequency'] = 10000

if sampling_method == "gridsearch": # choose the grid
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_numStartPoints'] = num_sample_points
    PEUQSE.UserInput.parameter_estimation_settings['multistart_initialPointsDistributionType'] = 'grid'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_searchType'] = 'getLogP'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearch_threshold_filter_coefficient'] = 2.0 #The lower this is, the more the points become filtered. It is not recommended to go below 2.0.
    
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearchSamplingInterval'] = param_std #This is for gridsearches and is in units of absolute intervals. By default, these intervals will be set to 1 standard deviaion each.  To changefrom the default, make a comma separated list equal to the number of parameters.
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearchSamplingRadii'] = [] #This is for gridsearches and refers to the number of points (or intervals) in each direction to check from the center. For example, 3 would check 3 points in each direction plus the centerpointn for a total of 7 points along that dimension. For a 3 parameter problem, [3,7,2] would check radii of 3, 7, and 2 for those parameters.
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True
    PEUQSE.UserInput.parameter_estimation_settings['multistart_checkPointFrequency'] = 10000

if sampling_method == "sobol":
    PEUQSE.UserInput.parameter_estimation_settings['multistart_initialPointsDistributionType'] = 'sobol'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_searchType'] = 'getLogP' # 2; possible options for uniform optimization: getLogP , doOptimizeLogP , doOptimizeSSR , doOptimizeNegLogP
    PEUQSE.UserInput.parameter_estimation_settings['multistart_numStartPoints'] = num_sample_points
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearch_threshold_filter_coefficient'] = 2.0 #The lower this is, the more the points become filtered. It is not recommended to go below 2.0.
    PEUQSE.UserInput.parameter_estimation_settings['multistart_relativeInitialDistributionSpread'] = 0.2 # the lower this is, the more concentrated it is
    PEUQSE.UserInput.parameter_estimation_settings['scaling_uncertainties_type'] = "off"
    PEUQSE.UserInput.parameter_estimation_settings['multistart_checkPointFrequency'] = 10000

if sampling_method == "astroidal":
    PEUQSE.UserInput.parameter_estimation_settings['multistart_initialPointsDistributionType'] = 'astroidal'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_searchType'] = 'getLogP' # 2; possible options for uniform optimization: getLogP , doOptimizeLogP , doOptimizeSSR , doOptimizeNegLogP
    PEUQSE.UserInput.parameter_estimation_settings['multistart_numStartPoints'] = num_sample_points
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearch_threshold_filter_coefficient'] = 2.0 #The lower this is, the more the points become filtered. It is not recommended to go below 2.0.
    PEUQSE.UserInput.parameter_estimation_settings['multistart_relativeInitialDistributionSpread'] = 0.2 # the lower this is, the more concentrated it is
    PEUQSE.UserInput.parameter_estimation_settings['scaling_uncertainties_type'] = "off"
    PEUQSE.UserInput.parameter_estimation_settings['multistart_checkPointFrequency'] = 10000

if sampling_method == "shell":
    PEUQSE.UserInput.parameter_estimation_settings['multistart_initialPointsDistributionType'] = 'shell'
    PEUQSE.UserInput.parameter_estimation_settings['multistart_searchType'] = 'getLogP' # 2; possible options for uniform optimization: getLogP , doOptimizeLogP , doOptimizeSSR , doOptimizeNegLogP
    PEUQSE.UserInput.parameter_estimation_settings['multistart_numStartPoints'] = num_sample_points
    # PEUQSE.UserInput.parameter_estimation_settings['multistart_exportLog'] = True
    PEUQSE.UserInput.parameter_estimation_settings['multistart_gridsearch_threshold_filter_coefficient'] = 2.0 #The lower this is, the more the points become filtered. It is not recommended to go below 2.0.
    PEUQSE.UserInput.parameter_estimation_settings['multistart_relativeInitialDistributionSpread'] = 0.2 # the lower this is, the more concentrated it is
    PEUQSE.UserInput.parameter_estimation_settings['scaling_uncertainties_type'] = "off"
    PEUQSE.UserInput.parameter_estimation_settings['multistart_checkPointFrequency'] = 10000