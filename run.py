from sys import argv
from os import system
from os.path import abspath

data_path = abspath(argv[1])
data_name = data_path.split('/')[-1]

lsf_fn = 'run_%s.lsf' %data_name
fn = open(lsf_fn,'w')
fn.write('#!/bin/bash\n#BSUB -J %s\n#BSUB -P acc_pandeg01a\n#BSUB -q expressalloc\n#BSUB -n 100\n#BSUB -W 6:00\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R rusage[mem=20480]\n' %(data_name,data_name,data_name))
fn.write('module load python\nmodule load py_packages\nmodule load java\nmodule load groovy\nmodule load selfsched\nmodule load selfsched\nmodule load weka\n')
fn.write('export _JAVA_OPTIONS="-XX:ParallelGCThreads=10"\nexport JAVA_OPTS="-Xmx10g"\nexport CLASSPATH=/hpc/users/wangl35/.sdkman/candidates/java/weka.jar\n')
fn.write('python prepare.py %s\n' %data_path)
fn.write('mpirun selfsched < %s.jobs\n' %data_name)
fn.close()

system('bsub < %s' %lsf_fn)
system('rm %s' %lsf_fn)
