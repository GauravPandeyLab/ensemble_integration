'''
Combine predictions of models using different feature sets.
Author: Linhua (Alex) Wang
Date:  12/27/2018
'''
from os.path import exists,abspath,isdir,dirname
from sys import argv
from os import listdir,environ
from common import load_properties
import pandas as pd
import numpy as np

data_folder = abspath(argv[1])
#data_name = data_folder.split('/')[-1]

fns = listdir(data_folder)
fns = [data_folder  + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]

bagValues = range(int(argv[2]))

prediction_dfs = []
validation_dfs = []

for value in bagValues:
	prediction_dfs = []
	validation_dfs = []
	for folder in feature_folders:
		feature_name = folder.split('/')[-1]
		prediction_df = pd.read_csv(folder + '/predictions-%d.csv.gz' %value,compression='gzip')
		prediction_df.set_index(['id','label'],inplace=True)
		prediction_df.columns = ['%s.%s' %(feature_name,col) for col in prediction_df.columns]

		validation_df = pd.read_csv(folder + '/validation-%d.csv.gz' %value,compression='gzip')
		validation_df.set_index(['id','label'],inplace=True)
		validation_df.columns = ['%s.%s' %(feature_name,col) for col in validation_df.columns]

		prediction_dfs.append(prediction_df)
		validation_dfs.append(validation_df)

	prediction_dfs = pd.concat(prediction_dfs,axis=1)
	validation_dfs = pd.concat(validation_dfs,axis=1)

	prediction_dfs.to_csv(data_folder + '/predictions-%d.csv.gz' %value,compression='gzip')
	validation_dfs.to_csv(data_folder + '/validation-%d.csv.gz' %value,compression='gzip')
