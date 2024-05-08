#%% imports
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import GPy
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import Exponentiation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import CheKiPEUQ as ch
from multiprocessing import Pool
import UserSettings

#%% functions
def split_data(df, test_split):
    """
    :param df: variable containing data for training/testing
    :param test_split: ratio of test data to training data
    :type df: DataFrame/nd array
    :type test_split: float
    :return train: full training data
    :return test: full testing data
    """

    training, testing = train_test_split(df,
                                         test_size=test_split,
                                         )
    training = training.reset_index(drop=True)
    testing = testing.reset_index(drop=True)
    return training, testing


def splitXY(training, testing, label_headers):
    """
    :param training: training data
    :param testing: testing data
    :return features_training: features for training
    :return features_testing: features for testing
    :return labels_training: labels/y-values for testing
    :return labels_testing: labels for testing
    """
    features_training = training[UserSettings.feature_headers].to_numpy()
    features_testing = testing[UserSettings.feature_headers].to_numpy()
    labels_training = training[label_headers].to_numpy()
    labels_testing = testing[label_headers].to_numpy()
    return features_testing, features_training, labels_training, labels_testing


# def plot_kde_kernel(title, predicted, true, bin_size=20):
#     plt.title(title)
#     sns.displot(true, hist=True, kde=False, norm_hist=True,
#                  bins=bin_size, hist_kws=dict(alpha=0.4))
#     try:
#         sns.kdeplot(data=predicted, cbar=True, kernel="gau",
#                     bw="scott", label="predicted", gridsize=1000)
#         sns.kdeplot(data=true, cbar=True, kernel="gau",
#                     bw="scott", label="true", gridsize=1000)
#     except UserWarning:  # statsmodels bypasses the kernel density estimation if bw < 0
#         sns.kdeplot(data=predicted, cbar=True, kernel="cos",
#                     bw="scott", label="predicted", gridsize=1000)
#         sns.kdeplot(data=true, cbar=True, kernel="cos",
#                     bw="scott", label="true", gridsize=1000)
#     plt.legend(loc=2,
#                prop={'size': 12})


def surrogate(x_train, x_test, y_train, y_test, kernel, normalize=True, print_details=False):
    try:
        x, y = y_train.shape
    except ValueError:
        y_train = y_train[:, np.newaxis]
    try:
        gpr = GPy.models.GPRegression(X=x_train,
                                    Y=y_train,
                                    kernel=kernel,
                                    normalizer=normalize)
        gpr.optimize()
        if print_details:
            print(gpr)
        predictions, likelihoods = gpr.predict(x_test)
        return [float(prediction) for prediction in predictions], \
            [float(likelihood) for likelihood in likelihoods]
    except:
        predictions = None
        likelihoods = None
        return predictions, likelihoods


# changed to output R2 value and GP Model
def final_surrogate(x_train, x_test, y_train, y_test, kernel, normalize=True, print_details=False):
    try:
        x, y = y_train.shape
    except ValueError:
        y_train = y_train[:, np.newaxis]
    try:
        gpr = GPy.models.GPRegression(X=x_train,
                                    Y=y_train,
                                    kernel=kernel,
                                    normalizer=normalize)
        gpr.optimize()
        if print_details:
            print(gpr)
        predictions, likelihoods = gpr.predict(x_test)
        r2 = r2_score(y_test, predictions)
        return gpr, r2, predictions
    except: 
        return None, None, None


def test_gp(gpr, x_test, y_test):
    predictions, likelihoods = gpr.predict(x_test)
    r2 = r2_score(y_test, predictions)
    return predictions, r2 

def gp_analysis(label):
    df = pd.read_csv(UserSettings.datafile_name, header=0)
    dim = len(UserSettings.feature_headers)
    df = df[UserSettings.feature_headers+[label]]
    df_full = df
    df = df[:1000]
    # type_sobol = ['0.25%', '0.5%', '1%', '2%', '4%'] 
    # df['type_sobol'] = '1%'
    # for num, type_ in enumerate(type_sobol):
    #     df['type_sobol'][num*1000:(num+1)*1000] = type_
    # df = df[UserSettings.feature_headers + ['type_sobol'] + [label]]
    # df = df[df[label]!='[]']
    df[UserSettings.feature_headers + [label]] = df[UserSettings.feature_headers + [label]].apply(pd.to_numeric)
    # dfp = df[df['type_sobol']==sobol_p]
    features = df[UserSettings.feature_headers].to_numpy() # all feature data
    labels = df[label].to_numpy() # all label data


    # df[UserSettings.feature_headers] = (df[UserSettings.feature_headers] - df[UserSettings.feature_headers].mean()) / df[UserSettings.feature_headers].std()
    test_split = UserSettings.training_data_percent
    train, test = split_data(df, test_split)
    features_test, features_train, labels_train, labels_test = splitXY(train, test, label)
    # analyze kernels
    all_titles = ['RBF', 'Exponential', 'Mat32', 'Mat52', 'Cosine', 'RBF+RBF'] 
    all_kerns = [GPy.kern.RBF(dim), GPy.kern.Exponential(dim), GPy.kern.Matern32(dim), GPy.kern.Matern52(dim),
                GPy.kern.Cosine(dim), GPy.kern.RBF(dim) + GPy.kern.RBF(dim)] # GPy.kern.Cosine(dim),
    titles = UserSettings.gp_kernels
    kerns = [all_kerns[i] for i, item in enumerate(all_titles) if item in titles]
     # make for user influence, use eval statement in a for loop with append to create kerns without having to initialize all the kernel objects
     # 'Cosine',
    
    
    kern_dict = {title: kern for kern, title in zip(
        kerns, titles)}  # Initializing dictionaries
    kernel_analysis = {"gp_predictions": {},
                    "true_values": {}, 'mse': {}, 'rr': {}}
    best_kernels_rr = {}
    # plt.subplots(3, 2, tight_layout=True, figsize=(
    #         15, 10))  
    for kern_num, kernel, title in zip(range(len(kerns)), kerns, titles):
        kern_title = title
        kernel_analysis["gp_predictions"][f"{label}"] = {}
        # Initialize Data
        gpy_features = features_train[:]
        gpy_test_features = features_test[:]
        gpy_labels = labels_train
        # Needs to be one dimensional column vector not 2-D vector -> eg. (13,) not (13,1)
        gpy_labels = gpy_labels[:, None]
        # ) not (13,1)
        actual = labels_test
        # Create Regressor object
        prediction, _ = surrogate(
            gpy_features, gpy_test_features, gpy_labels, actual, kernel)
        print("Optimizing", label, 'for', kern_title)
        # Plotting
        # plt.subplot(3, 2, kern_num + 1)  
        # plot_kde_kernel(kern_title, prediction, actual)

        # Scoring
        if prediction != None:
          # Plotting
        #   plt.subplot(3, 2, kern_num + 1)  
        #   plot_kde_kernel(kern_title, prediction, actual)

          # Scoring
          try:
            rr = r2_score(actual, prediction)
            ms_error = mean_squared_error(actual, prediction)
          except ValueError:
            rr = 0
            ms_error = 0
            print(f'{label} with {kern_title} has an error')

          # Property Analysis and Storage
          kernel_analysis["gp_predictions"][label][kern_title] = {
              'values_predicted': prediction}
          kernel_analysis["true_values"][label] = actual.tolist()
          kernel_analysis['mse'][kern_title] = ms_error
          kernel_analysis['rr'][kern_title] = rr
          plt.suptitle(f"{label}")
        else:
          print(f'{label} with {kern_title} has an error')

        # Property Analysis and Storage
    best_kernel = max(kernel_analysis["rr"].keys(), key=(
        lambda k: kernel_analysis["rr"][k]))
    best_rr = kernel_analysis["rr"][best_kernel]
    best_kernels_rr[label] = {best_kernel: best_rr}
    # save_path = f"label_plots/{label}.png"

    # plt.savefig(save_path, format="png")
    #plt.show()
    #plt.close()
    # Deleting temporary dictionaries
    del kernel_analysis["mse"]
    del kernel_analysis["gp_predictions"]
    del kernel_analysis['true_values']

    # TODO: Functionalize
    # Creating json dump files
    with open(f"Kern_Data/kernel_analysis_{label}.json", 'w') as file1, open(f"Kern_Data/best_kernels_{label}.json", "w") as file2:
        json_string = json.dumps(
            kernel_analysis, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file1.write(json_string)
        json_string = json.dumps(
            best_kernels_rr, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file2.write(json_string)
    print("Wrote values to JSON!")

    #Create Best GP
    path = os.getcwd()
    os.chdir(path + '/GP_models')
    # best_kernels = [kern_dict[kernel] for kernel in [best_kernel]]
    gp, rr, predictions = final_surrogate(features, features, labels, labels, kern_dict[best_kernel])
    if gp != None:
        # print(f'R2 value for {label}+{sobol_p} with {best_kernel} Kernel: {rr}')
        # sobol_p = sobol_p.replace('%', 'p')
        ch.dillpickleAnObject(gp, f'surrogate_{UserSettings.forcefield_type}_{UserSettings.metal_name}_{label}') 
        os.chdir('..')
        # # evaluate posterior models
        os.chdir('Comparison_Plots')
        # # sobol sections points graphs
        # os.chdir('Region_points')
        # plt.figure(figsize=(15,10))
        # plt.plot(predictions, labels)
        # plt.xlabel('Predicted')
        # plt.ylabel('True Values')
        # plt.title(f'{label} in {sobol_p} range')
        # plt.gcf().savefig(f'comparison_{sobol_p}_{label}.png')
        # plt.close()
        # os.chdir('..')
        # sobol secion compared to all points

        os.chdir('All_points')
        all_predict, all_r2 = test_gp(gp, df_full[UserSettings.feature_headers].to_numpy(), df_full[label].to_numpy())
        plt.figure(figsize=(15,10))
        plt.scatter(all_predict, df_full[label].to_numpy())
        print("This is df:", np.shape(df_full[label].to_numpy()))
        test=np.concatenate((all_predict, np.reshape(df_full[label].to_numpy(),(4335,1))),axis=1)
        np.savetxt(f'comparison_{label}.txt', test, delimiter=' ')
        plt.xlabel('Predicted')
        plt.ylabel('True Values')
        plt.title(f'{label}')
        plt.gcf().savefig(f'comparison_{label}.png')
        plt.close()
        os.chdir('..')
        r2_values = {f'{label}': ({"r2 small": rr}, {"r2 full": all_r2})} # all_r2
        os.chdir('..')
        return r2_values
    else: 
        r2_values = {f'{label}': {"r2": None}}
        return r2_values




#%% create folders
if __name__ == '__main__':
# creating folders
# folders = ['BGP_plots', 'GP_data', 'trace', 'LOOCV_plots', 'label_plots_pm']
    folders = ['Kern_Data', 'GP_models', 'label_plots', 'Comparison_Plots'] # TODO: Add to this as I go
    for folder in folders:
        try:
            os.mkdir(folder)
        except OSError as error:
            print(' ')
    os.chdir('Comparison_Plots')
    folders_sub = ['All_points'] # , 'Region_points'
    for folder in folders_sub:
        try:
            os.mkdir(folder)
        except OSError as error:
            print(' ')
    os.chdir('..')
#%% execution
    pool = Pool()
    # pickle r2_values and make 
    r2_values = pool.map(gp_analysis, UserSettings.label_headers)
    with open(f"Comparison_Plots/r2_values_{UserSettings.metal_name}.json", 'w') as file1:
        json_string = json.dumps(
            r2_values, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        file1.write(json_string)


### NOTES TO CHANGE ###
# make UserInput file to contain label_headers, UserSettings.feature_headers, etc. 
# UserSettings.py
# in bayesianPd.py, change sampling method as UserSettings and set flags to change the sampler method
