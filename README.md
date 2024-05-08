# BUQ_all-atom_gold

You need sobol data in a CSV file (Sobols input parameters and output properties - here sample file is AA_Au_UQ - new_set.csv). GPparallelgen.py uses this sobol data file to generate GPR models. Subsequently, GPR models are used to evaluate the samples produced during bayesian parameter estimation. This evaluation is needed to determine the likelihood of those samples which is used to determine the posterior probability.

Usersettings.py: (This is the most important file) Following are the details of each command that needs to be changed as per users requirement:

metal_name: This is the name of material you are modeling
forcefield_type: Change to AA or CG system (Note this is just nomenclature for files that will be generated)
label_headers: The names of properties exactly as mentioned in the sobol_data.csv
feature_headers: The names of parameters exactly as mentioned in the sobol_data.csv
features_math_expression: The names of parameters that you would like to put as symbols instead of words.
features_err_math_expression: Ignore this
datafile_name: This is the name of file containing sobol data with all parameters and properties i.e. Sobol_data.csv (Note: Remove the garbage values or clean this data before running GPparallelgen.py)
best_param_set: This is the best parameter set that you obtained from PSO or it can be any model that you have developed and want to perform BUQ upon.
label_uncertainty_percents: This can be ignored since you explicitly mention ‘label_uncertainty’
target_values: These are the target values that you wanted you model to predict. Simply this can be directly the values that you used as targets in PSO.
best_prop_set: These are the properties that your ‘best_param_set’ has predicted.
label_uncertainty: This is an  array of all properties’s experimental standard deviation. If you don’t have experimental data, and if you are developong CG models trying to mimic some other AA system, you can use the standard deviation that you have in AA system results. 
training_data_percent: This is the percentage of data that you have in Sobol_data.csv that you would like to use for testing.
num_points_training: This can be ignored
gp_kernels: These are the list of kernels that you’ll be using to train your models on.
cross_validation_type: I don’t think this is used anywhere.
percent_bound_extension: These are the extensions to the bounds given by sobol data in terms of percentage.
param_prior_type: Specify the prior knowledge on the parameters. This will be used to determine the log Pprior.
sampling_method: Specify what sampling method you would like BUQ to use. 
num_sample_points: These are the number of samples that your BUQ will generate. Note, when you run the bayesian.py the continuous bar shows the number of epochs given by num_sample_points/num_walkers. 


Steps to run Bayesian Parameter Estimation

Download and install PEUQSE: pip install PEUQSE[COMPLETE]

Develop GPR models: Python GPparallelgen.py

Run BPE: Python bayesian.py



