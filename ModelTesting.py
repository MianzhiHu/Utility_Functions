from ComputationalModeling import *
from DualProcess import *
from VisualSearchModels import *
import pandas as pd

if __name__ == "__main__":
    # # ==================================================================================================================
    # # Model Testing for the ABCD Task
    # # ==================================================================================================================
    # # Read example data
    # dataABCD = pd.read_csv('./Test Data/ABCD_ContRewards.csv')
    # dataABCD['choice'] = dataABCD['choice'] - 1  # Adjusting choice to be zero-indexed
    # dataABCD['setSeen'] = dataABCD['setSeen'] - 1  # Adjusting setSeen to be zero-indexed
    # dataABCD = dataABCD.drop(columns=['subjID']) # Remove the subjID column
    # testing_data = dict_generator(dataABCD[dataABCD['subnum'] == 1], task='ABCD')
    #
    # # model = ComputationalModels(model_type='delta', task='ABCD')
    # model = DualProcessModel(task='ABCD')
    # result = model.fit(testing_data, initial_EV=[0, 0, 0, 0], initial_mode='first_trial', num_iterations=1)

    # # ================================================================================================================
    # # Model Testing for IGT and SGT
    # # ================================================================================================================
    # # Read example data
    # dataIGT = pd.read_csv('./Test Data/IGT_ConcatDataManyLabs.csv')
    # dataIGTSGT = pd.read_csv('./Test Data/IGTSGT_OrderData.csv')
    #
    # # Clean the data
    # dataIGT['choice'] = dataIGT['choice'] - 1
    #
    # # Extract testing data
    # testing_data = dict_generator(dataIGT[dataIGT['Subnum'] == 1], task='IGT_SGT')
    #
    # # Load the model
    # model = ComputationalModels(model_type='delta', task='IGT_SGT', initial_EV=[0, 0, 0, 0], initial_mode='first_trial')
    # result = model.fit(testing_data, num_iterations=1)

    # ==================================================================================================================
    # Model Testing for the Visual Search Task
    # ==================================================================================================================
    # Read example data
    dataVS = pd.read_csv('./Test Data/LeSaS1_cleaned_data.csv')
    dataVS['Optimal_Choice'] = dataVS['Optimal_Choice'].astype(int)  # Ensure Optimal_Choice is integer type
    testing_data = dict_generator(dataVS[dataVS['SubNo'] == 1], task='VS')

    # Load the model
    # model = VisualSearchModels(model_type='delta', task='VS', initial_EV=[0, 0], initial_mode='first_trial')
    model = VisualSearchModels(model_type='hybrid_decay_delta_3', task='VS')
    result = model.fit(testing_data, initial_EV=[0, 0], initial_RT=[0, 0], initial_mode='fixed', num_iterations=1)

    # # ==================================================================================================================
    # # Model Testing for the Sliding Window Model Fitting Approach
    # # ==================================================================================================================
    # # Read example data
    # dataABCD = pd.read_csv('./Test Data/ABCD_ContRewards.csv')
    # dataABCD['choice'] = dataABCD['choice'] - 1  # Adjusting choice to be zero-indexed
    # dataABCD['setSeen'] = dataABCD['setSeen'] - 1  # Adjusting setSeen to be zero-indexed
    # dataABCD = dataABCD.drop(columns=['subjID']) # Remove the subjID column
    # testing_data = dataABCD[dataABCD['subnum'] <= 2]
    #
    # # only keep the first 20 rows for testing
    # testing_data = testing_data.groupby('subnum').head(20)
    # testing_dict = dict_generator(testing_data, task='ABCD')
    #
    # model = ComputationalModels(model_type='delta', task='ABCD')
    # non_sliding_window_model = model.fit(testing_dict, initial_EV=[0, 0, 0, 0], initial_mode='first_trial', num_iterations=1)
    # sliding_window_model = moving_window_model_fitting(testing_data, model, task='ABCD', id_col='subnum',
    #                                                 num_iterations=1, window_size=10, restart_EV=False)
    #
    # x = non_sliding_window_model['best_EV']
    # print("Non-Sliding Window Model Best EV:", x)