from os.path import abspath, isdir
from os import remove, system, listdir
import argparse
from common import load_properties
from itertools import product
from os import environ, system
from os.path import abspath, dirname, exists
from sys import argv
from common import load_arff_headers, load_properties
from time import time

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

### parse arguments
parser = argparse.ArgumentParser(description='Feed some bsub parameters')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
parser.add_argument('--node', '-N', type=str, default='20', help='number of node requested')
parser.add_argument('--time', '-T', type=str, default='30:00', help='number of hours requested')
parser.add_argument('--memory', '-M', type=str,default='10240', help='memory requsted in MB')
parser.add_argument('--classpath', '-CP', type=str,default='./weka.jar', help='default weka path')
parser.add_argument('--minerva', type=str2bool,default='true', help='use minerva cluster or not')
parser.add_argument('--fold', '-F', default='5', help='cross-validation fold')
args = parser.parse_args()
### record starting time
start = time()
### get the data path
data_path = abspath(args.path)
data_name = data_path.split('/')[-1]
working_dir = dirname(abspath(argv[0]))

### get weka properties from weka.properties
p = load_properties(data_path)
fold_values = range(int(p['foldCount']))
bag_values = range(int(p['bagCount']))

### get the list of base classifiers
classifiers_fn = data_path + '/classifiers.txt'
assert exists(classifiers_fn) 
classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
classifiers = [_.strip() for _ in classifiers]

### get paths of the list of features
fns = listdir(data_path)
fns = [data_path  + '/' + fn for fn in fns]
feature_folders = [fn for fn in fns if isdir(fn)]
assert len(feature_folders) > 0

### write the individual tasks
classpath = args.classpath
all_parameters = list(product(feature_folders, classifiers,fold_values,bag_values))
job_file = open('%s.jobs' %data_name,'w')
for parameters in all_parameters:
	project_path, classifier, fold, bag = parameters
	job_file.write('groovy -cp %s %s/base_model.groovy %s %s %s %s %s\n' % (classpath, working_dir,data_path, project_path, fold, bag,classifier))
if not args.minerva:
	job_file.write('python combine_individual_feature_preds.py %s\npython combine_feature_predicts.py %s %s\n' %(data_path,data_path,args.fold))
job_file.close()

### submit to minerva if args.minerva != False
if args.minerva:
	lsf_fn = 'run_%s.lsf' %data_name
	fn = open(lsf_fn,'w')
	fn.write('#!/bin/bash\n#BSUB -J EI-%s\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -n %s\n#BSUB -sp 100\n#BSUB -W %s\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R rusage[mem=20000]\n' %(data_name,args.queue,args.node,args.time,data_name,data_name))
	fn.write('module load python/2.7.14\nmodule load py_packages\nmodule load java\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
	fn.write('export _JAVA_OPTIONS="-XX:ParallelGCThreads=10"\nexport JAVA_OPTS="-Xmx15g"\nexport CLASSPATH=%s\n' %(args.classpath))
	#fn.write('python prepare.py %s\n' %data_path)
	fn.write('mpirun selfsched < %s.jobs\n' %data_name)
	fn.write('rm %s.jobs\n' %data_name)
	fn.write('python combine_individual_feature_preds.py %s\npython combine_feature_predicts.py %s %s\n' %(data_path,data_path,args.fold))
	fn.close()
	system('bsub < %s' %lsf_fn)
	system('rm %s' %lsf_fn)

### run it sequentially otherwise
else:
	system('sh %s.jobs' %data_name)
	system('rm %s.jobs' %data_name)
end = time()
if not args.minerva:
	print 'Elapsed time is: %s seconds' %(end -start)
