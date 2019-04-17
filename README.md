## Configurations

1. Install Java and groovy.

This can be done using sdkman (https://sdkman.io/).

2. Install python environments using conda.

	conda create --name ei
	conda install -n ei python=2.7.14 cython=0.19.1 pandas scikit-learn

3. Download weka.jar to the current directory.

	curl -O -L http://prdownloads.sourceforge.net/weka/weka-3-7-10.zip

## Data

Under the data path, 2 files and a list of feature folders are expected: 

1. classifiers.txt
This file specifies the list of base classifiers. See the sample_data/classifiers.txt as an example.

2. weka.properties
This file specifies the list of weka properties that are parsed to the training/testing of base classifiers. See the sample_data/weka.properties as an example.

3. Folders for feature sets
This is a list of folders under the main data path. Each of them originally contains only one file named as data.arff. The .arff files are the input feature matrices and labels for training/testing Weka base classifiers. Indices and labels of .arff files should be aligned across all feature sets. 

## Run the pipeline

### Train base classifiers

Arguments of train_base.py:
--path: Path to data, required.
--minerva: Specify whether use minerva or not, default is true. Following parameters work only when minerva is requested. 
	--queue: minerva queue to be used, default is premium. 
	--node: #nodes to be used, default is 20. 
	--time: walltime, default is 10:00 (10 hours).
	--memory: memory limit, default is 10240 MB.
	--classpath: path to weka file, default is ./weka.jar.

Option 1: Without access to Minerva, EI can be run sequentially.

	python train_base.py --path [path] --minerva 0 

Option 2: Run the pipeline in parallel on Minerva

	python train_base.py --path [path] --node [#node] --queue [queue] --time [hour:min] --memory [memory]

### Train and evaluate EI

Arguments of ensemble.py:
--path: path to data, required.
--fold: cross validation fold, default is 5.
--agg: number to aggregate bag predictions, default is 1. 
	If each bagging classifier is treated individually, ignore it. Else if bagcount > 1, set -agg = bagcount. 
	Run the follwoing command:

	python ensemble.py --path P --fold fold_count --agg agg_number

F-max scores of these models will be printed and written in the performance.csv file and saved to the analysis folder under the data path.