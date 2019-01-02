'''
Prepare independent tasks for training base classifiers.
Author: Linhua (Alex) Wang
Date:  12/26/2018
'''
from itertools import product
from os.path import exists,abspath,isdir,dirname
from sys import argv
from os import listdir,environ
from common import load_properties

working_dir = dirname(abspath(argv[0]))
data_folder = abspath(argv[1])
data_name = data_folder.split('/')[-1]


p = load_properties(data_folder)
fold_values = range(int(p['foldCount']))
bag_values = range(int(p['bagCount']))

classifiers_fn = data_folder + '/classifiers.txt'
assert exists(classifiers_fn) 
classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
classifiers = [_.strip() for _ in classifiers]

fns = listdir(data_folder)

fns = [data_folder  + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]

assert len(feature_folders) > 0

classpath = environ['CLASSPATH']


all_parameters = list(product(feature_folders, classifiers,fold_values,bag_values))


job_file = open('%s.jobs' %data_name,'w')
for parameters in all_parameters:
	project_path, classifier, fold, bag = parameters
 	job_file.write('groovy -cp %s %s/base.groovy %s %s %s %s.arff %s\n' % (classpath, working_dir, project_path, fold, bag,data_name, classifier))

job_file.close()

