#TODO: load local models
import argparse
import pickle
import os
import pandas
import pandas as pd

from processing_scripts.common import load_properties, str2bool
from time import time
from os.path import abspath, dirname, exists, isdir
from sys import argv
from os import listdir
from itertools import product
from os import system


def base_predictors(model_path, data_path, hpc, classpath):
    start = time()
    # model_source_dir = model_path.split('/')[-2]
    # model_name = model_path.split('/')[-1]
    data_source_dir = data_path.split('/')[-2]
    data_name = data_path.split('/')[-1]
    working_dir = dirname(abspath(argv[0]))

    ### get weka properties from weka.properties
    p = load_properties(model_path)
    bag_values = range(int(p['bagCount']))

    ### get the list of base classifiers
    classifiers_fn = model_path + '/classifiers.txt'
    assert exists(classifiers_fn)
    classifiers = filter(lambda x: not x.startswith('#'), open(classifiers_fn).readlines())
    classifiers = [_.strip().split(' ')[0] for _ in classifiers]
    print(classifiers)

    ### get paths of the list of features
    # print(model_path)
    fns = listdir(model_path)
    # print(fns)
    excluding_folder = ['analysis']
    fns = [fn for fn in fns if not fn in excluding_folder]
    fns = [fn for fn in fns if not 'tcca' in fn]
    fns = [model_path + '/' + fn for fn in fns]
    model_feature_folders = [fn for fn in fns if isdir(fn)]

    fns = listdir(data_path)
    excluding_folder = ['analysis', 'feature_rank', 'model_built']
    fns = [fn for fn in fns if not fn in excluding_folder]
    fns = [fn for fn in fns if not 'tcca' in fn]
    fns = [data_path + '/' + fn for fn in fns]
    data_feature_folders = [fn for fn in fns if isdir(fn)]

    data_model_feat_list = []
    for fn in model_feature_folders:
        data_model_pair = [fn]
        for dfn in data_feature_folders:
            if fn.split('/')[-1] == dfn.split('/')[-1]:
                data_model_pair.append(dfn)
        data_model_feat_list.append(data_model_pair)

    print(data_model_feat_list)
    # get fold, id and label attribute

    fold_values = ['test']
    jobs_fn = "temp_train_base_{}_{}.jobs".format(data_source_dir, data_name)
    job_file = open(jobs_fn, 'w')
    if not hpc:
        job_file.write('module load groovy\n')

    def preprocessing(jf):
        # classpath = classpath
        all_parameters = list(product(data_model_feat_list, classifiers, fold_values, bag_values))

        for parameters in all_parameters:
            data_model_pair, classifier, fold, bag = parameters
            local_model_path, data_feat_dir = data_model_pair
            jf.write('groovy -cp %s %s/groovy_scripts/load_base_predictors.groovy %s %s %s %s %s %s\n' % (classpath, working_dir,
                                                                                         data_path,
                                                                                         data_feat_dir,
                                                                                         fold,
                                                                                         bag,
                                                                                         classifier,
                                                                                         local_model_path
                                                                                      ))

        if not hpc:
            jf.write('python processing_scripts/combine_individual_feature_preds.py %s %s %s\npython processing_scripts/combine_feature_predicts.py %s %s %s\n' % (
                data_path, 'False', 'True',
                data_path, 'False', 'True'))

        return jf

    job_file = preprocessing(job_file)
    job_file.close()

    ### submit to hpc if args.hpc != False
    if hpc:
        lsf_fn = 'run_%s_%s.lsf' % (data_source_dir, data_name)
        fn = open(lsf_fn, 'w')
        fn.write(
            '#!/bin/bash\n#BSUB -J EI-%s\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -n %s\n#BSUB -W %s\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R rusage[mem=%s]\n' % (
                # '#!/bin/bash\n#BSUB -J EI-%s\n#BSUB -P acc_pandeg01a\n#BSUB -q %s\n#BSUB -n %s\n#BSUB -W %s\n#BSUB -o %s.stdout\n#BSUB -eo %s.stderr\n#BSUB -R himem\n' % (
                data_name, args.queue, args.node, args.time, data_source_dir, data_source_dir, args.memory))
        fn.write('module load java\nmodule load python\nmodule load groovy\nmodule load selfsched\nmodule load weka\n')
        fn.write(
            'export _JAVA_OPTIONS="-XX:ParallelGCThreads=10"\nexport JAVA_OPTS="-Xmx{}g"\nexport CLASSPATH={}\n'.format(
                int(float(args.memory) / 1024) - 1, args.classpath))
        fn.write('mpirun selfsched < {}\n'.format(jobs_fn))
        fn.write('rm {}\n'.format(jobs_fn))
        fn.write('python processing_scripts/combine_individual_feature_preds.py %s %s %s\npython processing_scripts/combine_feature_predicts.py %s %s %s\n' % (
                data_path, 'False', 'True',
                data_path, 'False', 'True'))
        fn.close()
        system('bsub < %s' % lsf_fn)
        system('rm %s' % lsf_fn)

    ### run it sequentially otherwise
    else:
        system('sh %s' % jobs_fn)
        system('rm %s' % jobs_fn)
    end = time()
    if not args.hpc:
        print('Elapsed time is: %s seconds' % (end - start))

    # return local_predictions


def ensemble(model_path, data_path, ens_model, regression=False):
    ens_model_path = os.path.join(model_path, 'analysis/ens_model.pkl')
    ens_model_dict = pickle.load(open(ens_model_path,'rb'))
    data_df = pandas.read_csv(os.path.join(data_path, 'predictions-test.csv.gz'), index_col=0)
    # print(data_df)
    if ens_model == "Mean":
        ens_prediction_np_array = data_df.mean(axis=1).values
    elif ens_model == "CES":
        ces_combination = ens_model_dict[ens_model][0]
        # print(ces_combination)
        ces_comb_bag = [c+'.0' for c in ces_combination.tolist()]
        ces_bp_df = data_df[ces_comb_bag]
        ens_prediction_np_array = ces_bp_df.mean(axis=1).values
    elif 'S.' in ens_model:
        stacker = ens_model_dict[ens_model]
        if hasattr(stacker, "predict_proba") and (not regression):
            ens_prediction_np_array = stacker.predict_proba(data_df)[:, 1]
        else:
            ens_prediction_np_array = stacker.predict(data_df)
            if regression is False:
                ens_prediction_np_array = ens_prediction_np_array[:, 1]
    ens_prediction = pd.DataFrame({'id': data_df.index,
                        'prediction': ens_prediction_np_array})
    print(ens_prediction)
    ens_prediction.to_csv(os.path.join(data_path, 'prediction_score.csv'))


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    # fmax_sklearn = make_scorer(common.f_max, greater_is_better=True, needs_proba=True)
    # auprc_sklearn = make_scorer(common.auprc, greater_is_better=True, needs_proba=True)
    ### parse arguments
    parser = argparse.ArgumentParser(description='Ensemble script of EI')
    parser.add_argument('--model_path', '-mp', type=str, required=True, help='Path of the EI model')
    parser.add_argument('--data_path', '-dp', type=str, required=True, help='Path of the multimodal data')
    # parser.add_argument('--fold', '-F', type=int, default=5, help='cross-validation fold')
    parser.add_argument('--aggregate', '-A', type=int, default=1, help='if aggregate is needed, feed bagcount, else 1')
    parser.add_argument('--hpc', type=str2bool, default='True', help='Boolean of using HPC to compute (default:True)')
    parser.add_argument('--queue', '-Q', type=str, default='premium', help='LSF queue to submit the job')
    parser.add_argument('--node', '-N', type=str, default='32', help='number of node requested')
    parser.add_argument('--time', '-T', type=str, default='40:00', help='number of hours requested')
    parser.add_argument('--memory', '-M', type=str, default='16000', help='memory requsted in MB')
    parser.add_argument('--classpath', '-CP', type=str, default='./weka.jar', help='default weka path')
    parser.add_argument('--rank', type=str2bool, default='False', help='Boolean of getting local model ranking or not (default:False)')
    parser.add_argument('--local_predictor', type=str2bool, default='False', help='Boolean of loading local_models (default:False)')
    parser.add_argument('--ens_model', type=str, default='Choose one of the ensemble', help='Choose the ensemble for EI interpretation')

    args = parser.parse_args()

    if args.local_predictor:
        base_predictors(args.model_path, args.data_path, args.hpc, args.classpath)
    else:
        ensemble(args.model_path, args.data_path, args.ens_model)

#TODO: load ensembles