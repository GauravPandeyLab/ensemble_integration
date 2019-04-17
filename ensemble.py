import common
import pandas as pd
import argparse
from time import time
from os import mkdir
from os.path import abspath, exists
from sys import argv
from numpy import array, column_stack, append
from numpy.random import choice, seed
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.linear_model import SGDClassifier # SGD
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.linear_model import LogisticRegression # Logistic regression
from sklearn.ensemble import AdaBoostClassifier #Adaboost
from sklearn.tree import DecisionTreeClassifier # Decision Tree
from sklearn.ensemble import GradientBoostingClassifier # Logit Boost with parameter(loss='deviance')
from sklearn.neighbors import KNeighborsClassifier # K nearest neighbors (IBk in weka)
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

def checkFolder(path,fold_count=5):
    for fold in range(fold_count):
        if not exists('%s/predictions-%d.csv.gz' %(path,fold)):
            return False
        if not exists('%s/validation-%d.csv.gz' %(path,fold)):
            return False
    return True

def get_performance(df, ensemble, fold, seedval):
    labels          = df.index.get_level_values('label').values
    predictions     = df[ensemble].mean(axis = 1)
    return {'fold': fold, 'seed': seedval, 'score': common.fmax_score(labels, predictions), 'ensemble': ensemble[-1], 'ensemble_size': len(ensemble)}


def get_predictions(df, ensemble, fold, seedval):
    ids             = df.index.get_level_values('id')
    labels          = df.index.get_level_values('label')
    predictions     = df[ensemble].mean(axis = 1)
    diversity       = common.diversity_score(df[ensemble].values)
    return pd.DataFrame({'fold': fold, 'seed': seedval, 'id': ids, 'label': labels, 'prediction': predictions, 'diversity': diversity, 'ensemble_size': len(ensemble)})


def select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i):
    initial_ensemble_size = 2
    max_candidates=50
    if len(ensemble) >= initial_ensemble_size:
        candidates = choice(best_classifiers.index.values, min(max_candidates, len(best_classifiers)), replace = False)
        candidate_scores = [common.score(train_labels, train_df[ensemble + [candidate]].mean(axis = 1)) for candidate in candidates]
        best_candidate = candidates[common.argbest(candidate_scores)]
    else:
        best_candidate = best_classifiers.index.values[i]
    return best_candidate

def selection(fold,seedval,path,agg):
    seed(seedval)
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    train_df = common.unbag(train_df, agg)
    test_df = common.unbag(test_df, agg)
    best_classifiers = train_df.apply(lambda x: common.fmax_score(train_labels, x)).sort_values(ascending = not common.greater_is_better)
    train_performance = []
    test_performance = []
    ensemble = []
    for i in range(min(max_ensemble_size, len(best_classifiers))):
        best_candidate = select_candidate_enhanced(train_df, train_labels, best_classifiers, ensemble, i)
        ensemble.append(best_candidate)
        train_performance.append(get_performance(train_df, ensemble, fold, seedval))
        test_performance.append(get_performance(test_df, ensemble, fold, seedval))
    train_performance_df = pd.DataFrame.from_records(train_performance)
    best_ensemble_size = common.get_best_performer(train_performance_df).ensemble_size.values
    best_ensemble = train_performance_df.ensemble[:best_ensemble_size.item(0) + 1]
    return get_predictions(test_df, best_ensemble, fold, seedval), pd.DataFrame.from_records(test_performance)

def CES_fmax(path,fold_count=5,agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    method = 'enhanced'
    select_candidate = eval('select_candidate_' + method)
    method_function = selection
    initial_ensemble_size = 2
    max_ensemble_size = 50
    max_candidates = 50
    max_diversity_candidates = 5
    accuracy_weight = 0.5
    max_clusters = 20
    predictions_dfs = []
    performance_dfs = []
    seeds = range(agg)

    for seedval in seeds:
        for fold in range(fold_count):
            pred_df, perf_df = method_function(fold,seedval,path,agg)
            predictions_dfs.append(pred_df)
            performance_dfs.append(perf_df)
    performance_df = pd.concat(performance_dfs)
    performance_df.to_csv('%s/analysis/selection-%s-%s-iterations.csv' % (path, method, 'fmax'), index = False)
    predictions_df = pd.concat(predictions_dfs)
    predictions_df['method'] = method
    predictions_df['metric'] = 'fmax'
    predictions_df.to_csv('%s/analysis/selection-%s-%s.csv' % (path, method, 'fmax'), index = False)
    fmax = '%.3f' %(common.fmax_score(predictions_df.label,predictions_df.prediction))
    return float(fmax)

def mean_fmax(path,fold_count=5,agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []
    for fold in range(fold_count):
        _,_,test_df,label = common.read_fold(path,fold)
        test_df = common.unbag(test_df, agg)
        predict = test_df.mean(axis=1).values
        predictions = append(predictions,predict)
        labels = append(labels,label)
    fmax = '%.3f' %(common.fmax_score(labels,predictions))
    return float(fmax)

def bestbase_fmax(path,fold_count=5,agg=1):
    assert exists(path)
    if not exists('%s/analysis' % path):
        mkdir('%s/analysis' % path)
    predictions = []
    labels = []
    for fold in range(fold_count):
        _,_,test_df,label = common.read_fold(path,fold)
        test_df = common.unbag(test_df, agg)
        predictions.append(test_df)
        labels = append(labels,label)
    predictions = pd.concat(predictions)
    fmax_list = [common.fmax_score(labels,predictions.iloc[:,i]) for i in range(len(predictions.columns))]
    return max(fmax_list)

def stacked_generalization(path,stacker_name,stacker,fold,agg):
    train_df, train_labels, test_df, test_labels = common.read_fold(path, fold)
    test_df = common.unbag(test_df,agg)
    try:
        test_predictions = stacker.fit(train_df, train_labels).predict_proba(test_df)[:, 1]
    except:
        test_predictions = stacker.fit(train_df,train_labels).predict(test_df)[:,1]
    df = pd.DataFrame({'fold': fold, 'id': test_df.index.get_level_values('id'), 'label': test_labels, 'prediction': test_predictions, 'diversity': common.diversity_score(test_df.values)})
    return df

def main(path,fold_count=5,agg=1):
    dn = abspath(path).split('/')[-1]
    cols = ['data_name','fmax','method']
    dfs = []
    print '[CES] Start building model #################################'
    ces = CES_fmax(path,fold_count,agg)
    print '[CES] Finished evaluating model ############################'
    print '[CES] F-max score is %s.' %ces
    print '[Mean] Start building model ################################'
    mean = mean_fmax(path,fold_count,agg)
    print '[Mean] Finished evaluating model ###########################'
    print '[Mean] F-max score is %s.' %mean
    print '[Best Base] Start building model ###########################'
    bestbase = bestbase_fmax(path,fold_count,agg)
    print '[Best Base] Finished evaluating model ######################'
    print '[Best Base] F-max score is %s.' %bestbase
    dfs.append(pd.DataFrame(data = [[dn,ces,'CES']],columns=cols,index = [0]))
    dfs.append(pd.DataFrame(data = [[dn,mean,'Mean']],columns=cols,index = [0]))
    dfs.append(pd.DataFrame(data = [[dn,bestbase,'best base']],columns=cols,index = [0]))
    # Get Stacking Fmax scores
    stackers = [RandomForestClassifier(n_estimators = 200, max_depth = 2, bootstrap = False, random_state = 0), SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',probability=True,
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False),GaussianNB(),LogisticRegression(),AdaBoostClassifier(),DecisionTreeClassifier(),GradientBoostingClassifier(loss='deviance'),KNeighborsClassifier()]
    stacker_names = ["RF.S","SVM.S","NB.S","LR.S","AB.S","DT.S","LB.S","KNN.S"]
    for i,(stacker_name,stacker) in enumerate(zip(stacker_names,stackers)):
        print '[%s] Start building model ################################' %(stacker_name)
        predictions_dfs = [stacked_generalization(path,stacker_name,stacker,fold,agg) for fold in range(fold_count)]
        predictions_df = pd.concat(predictions_dfs)
        fmax = common.fmax_score(predictions_df.label, predictions_df.prediction)
	print '[%s] Finished evaluating model ###########################' %(stacker_name)
        print '[%s] F-max score is %s.' %(stacker_name,fmax)
	df = pd.DataFrame(data = [[dn,fmax,stacker_name]],columns=cols, index = [0])
        dfs.append(df)
    dfs = pd.concat(dfs)
    # Save results
    print 'Saving results #############################################'
    if not exists('%s/analysis' %path):
        mkdir('%s/analysis' %path)
    dfs.to_csv("%s/analysis/performance.csv" %path, index = False)


### parse arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', '-P', type=str, required=True, help='data path')
parser.add_argument('--fold', '-F', type=int,default=5, help='cross-validation fold')
parser.add_argument('--aggregate', '-A', type=int,default=1, help='if aggregate is needed, feed bagcount, else 1')
args = parser.parse_args()
main(args.path,args.fold,args.aggregate)

