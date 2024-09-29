from ComputationalModeling import *
import pandas as pd

# Read example data
dataIGT = pd.read_csv('./Test Data/IGT_ConcatDataManyLabs.csv')
dataIGTSGT = pd.read_csv('./Test Data/IGTSGT_OrderData.csv')

# Clean the data
dataIGT['choice'] = dataIGT['choice'] - 1

# Extract testing data
testing_data = dict_generator(dataIGT[dataIGT['Subnum'] == 1], task='IGT_SGT')

# Load the model
model = ComputationalModels(model_type='delta', task='IGT_SGT', num_trials=100)
result = model.fit(testing_data)

