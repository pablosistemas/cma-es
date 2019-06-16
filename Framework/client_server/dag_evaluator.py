import inspect
import joblib
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn import preprocessing

import custom_models
import dag_parser
import utils
from available_metrics import available_metrics

cache_dir = 'cache'

memory = joblib.Memory(cachedir=cache_dir, verbose=False)


# @memory.cache
def fit_model(model, values, targets, sample_weight=None):
    isClass = isinstance(model, ClassifierMixin)
    isRegr = isinstance(model, RegressorMixin)
    isSpl = isinstance(model, custom_models.KMeansSplitter)
    if isClass or isRegr or isSpl:
        if 'sample_weight' in inspect.signature(model.fit).parameters:
            return model.fit(values, targets, sample_weight=sample_weight)
    return model.fit(values, targets)


def data_ready(req, cache):
    """
    Checks that all required data are in the data_cache

    :param req: string or list of string containing the keys of
    required data in cache
    :param cache: dictionary with the computed data
    :return: Boolean indicating whether all required data are in cache
    """
    if not isinstance(req, list):
        req = [req]
    return all([r in cache for r in req])


def get_data(data_list, data_cache):
    """
    Gets the data specified by the keys in the data_list from the data_cache

    :param data_list: string or list of strings
    :param data_cache: dictionary containing the stored data
    :return: a single pandas.DataFrame if input is a string,
    a list of DataFrames if the input is a list of strings
    """
    if not isinstance(data_list, list):
        data_list = [data_list]
    tmp = [data_cache[d] for d in data_list]
    if len(tmp) == 1:
        return tmp[0]
    res = ([t[0] for t in tmp], [t[1] for t in tmp])
    return res


def append_all(data_frames):
    if not isinstance(data_frames, list):
        return data_frames
    res = data_frames[0]
    for i in range(1, len(data_frames)):
        res.append(data_frames[i])
    return res


def train_dag(dag, train_data, sample_weight=None):
    models = dict()
    data_cache = dict()

    # happens inside booster
    isarray_0 = isinstance(train_data[0], np.ndarray)
    isarray_1 = isinstance(train_data[1], np.ndarray)
    if isarray_0 and isarray_1:
        train_data = (pd.DataFrame(train_data[0]),
                      pd.Series(train_data[1]))

    data_cache[dag['input'][2]] = train_data
    models['input'] = True

    def unfinished_models():
        return [m for m in dag if m not in models]

    def data_available():
        return [m for m in dag if data_ready(dag[m][0], data_cache)]

    def next_methods():
        return [m for m in unfinished_models() if m in data_available()]

    while next_methods():

        for m in next_methods():
            # obtain the data
            features, targets, *rest = get_data(dag[m][0], data_cache)
            if rest:
                sample_weight = rest[0]
            ModelClass, model_params = utils.get_model_by_name(dag[m][1])

            for p in model_params:
                if model_params[p] == 'True':
                    model_params[p] = True
                elif model_params[p] == 'False':
                    model_params[p] = False
                elif model_params[p] == 'None':
                    model_params[p] = None

            out_name = dag[m][2]
            if dag[m][1][0] == 'stacker':
                sub_dags, initial_dag, input_data = \
                    dag_parser.extract_subgraphs(dag, m)
                model_params = dict(sub_dags=sub_dags, initial_dag=initial_dag)
                model = ModelClass(**model_params)
                features, targets = data_cache[input_data]
            elif isinstance(out_name, list):
                model = ModelClass(len(out_name), **model_params)
            else:
                model = ModelClass(**model_params)

            # build the model
            # some models cannot handle cases with only one class, we also need to check we are not working with a list
            # of inputs for an aggregator
            if custom_models.is_predictor(model) and isinstance(targets, pd.Series) and len(targets.unique()) == 1:
                model = custom_models.ConstantModel(targets.iloc[0])
            models[m] = fit_model(model, features, targets,
                                  sample_weight=sample_weight)
            # needed to update model if the result was cached
            model = models[m]

            # use the model to process the data
            if isinstance(model, custom_models.Stacker):
                data_cache[out_name] = model.train, targets.ix[model.train.index]
                continue
            if isinstance(model, custom_models.Aggregator):
                data_cache[out_name] = model.aggregate(features, targets)
                continue
            if custom_models.is_transformer(model):
                trans = model.transform(features)
            else:              # this is a classifier not a preprocessor
                trans = features                # the data do not change
                if isinstance(features, pd.DataFrame):
                    targets = pd.Series(
                        list(model.predict(features)), index=features.index)
                else:  # this should happen only inside booster
                    targets = pd.Series(list(model.predict(features)))

            # save the outputs
            # the previous model divided the data into several data-sets
            if isinstance(trans, list):
                if isinstance(model, custom_models.KMeansSplitter) and sample_weight is not None:
                    trans = [(x, targets.loc[x.index], sample_weight[model.weight_idx[i]])
                             for i, x in enumerate(trans)]  # need to divide the targets and the weights
                else:
                    trans = [(x, targets.loc[x.index])
                             for x in trans]     # need to divide the targets
                for i in range(len(trans)):
                    # save all the data to the cache
                    data_cache[out_name[i]] = trans[i]
            else:
                if isinstance(features, pd.DataFrame):
                    # we have only one output, can be numpy array
                    trans = pd.DataFrame(trans, index=features.index)
                else:
                    trans = pd.DataFrame(trans)
                trans.dropna(axis='columns', how='all', inplace=True)
                data_cache[out_name] = (trans, targets)    # save it

    return models


def test_dag(dag, models, test_data, output='preds_only'):
    data_cache = dict()
    finished = dict()

    if isinstance(test_data[0], np.ndarray):
        test_data = (pd.DataFrame(test_data[0]), test_data[1])

    if isinstance(test_data[1], np.ndarray):
        test_data = (test_data[0], pd.Series(
            test_data[1], index=test_data[0].index))

    data_cache[dag['input'][2]] = test_data
    finished['input'] = True

    def unfinished_models():
        return [m for m in dag if m not in finished]

    def data_available():
        return [m for m in dag if data_ready(dag[m][0], data_cache)]

    def next_methods():
        return [m for m in unfinished_models() if m in data_available()]

    while next_methods():

        for m in next_methods():

            # obtain the data
            features, targets = get_data(dag[m][0], data_cache)
            model = models[m]
            out_name = dag[m][2]

            # we got empty dataset (after same division)
            if isinstance(features, pd.DataFrame) and features.empty:
                # and we should divide it further
                if isinstance(out_name, list):
                    for o in out_name:
                        data_cache[o] = (features, targets)
                else:
                    data_cache[out_name] = (features, targets)
                finished[m] = True
                continue

            # use the model to process the data
            if isinstance(model, custom_models.Aggregator):
                data_cache[out_name] = model.aggregate(features, targets)
                finished[m] = True
                continue
            elif custom_models.is_transformer(model):
                trans = model.transform(features)
                targets = pd.Series(targets, index=features.index)
            else:                                                       # this is a classifier not a preprocessor
                trans = features                                        # the data do not change
                if isinstance(features, pd.DataFrame):
                    targets = pd.Series(
                        list(model.predict(features)), index=features.index)
                else:
                    targets = pd.Series(list(model.predict(features)))

            # save the outputs
            # the previous model divided the data into several data-sets
            if isinstance(trans, list):
                trans = [(x, targets.loc[x.index])
                         for x in trans]      # need to divide the targets
                for i in range(len(trans)):
                    # save all the data to the cache
                    data_cache[out_name[i]] = trans[i]
            else:
                if isinstance(features, pd.DataFrame):
                    # we have only one output, can be numpy array
                    trans = pd.DataFrame(trans, index=features.index)
                else:
                    trans = pd.DataFrame(trans)
                trans.dropna(axis='columns', how='all', inplace=True)
                data_cache[out_name] = (
                    trans, targets)                 # save it

            finished[m] = True

    if output == 'all':
        return data_cache['output']
    if output == 'preds_only':
        return data_cache['output'][1]
    if output == 'feats_only':
        return data_cache['output'][0]

    raise AttributeError(output, 'is not a valid output type')


input_cache = {}


def eval_metrics(metrics_dict, scores_dict, preds, targets):
    for metric, config_dict in metrics_dict.items():
        method = config_dict['method']
        args = config_dict['args']
        score = method(targets, preds, **args)

        scores_dict[metric].append(score)

    return scores_dict


def eval_dag_train_test(train_data, dag, metrics_dict, test_size, train_size):
    feats, targets = train_data

    scores_dict = {metric: [] for metric in metrics_dict.keys()}
    result = {}

    if test_size <= 0 or test_size >= 1:
        result = {'error': {
            'type': 'splits_param',
            'msg': 'If argument "splits" is float, it must be > 0 and < 1.'
        }}
        return result

    try:
        feats_train, feats_val, targets_train, targets_val = train_test_split(
            feats, targets, test_size=test_size, random_state=1)

        feats_train = feats_train[:int(train_size*len(feats_train))]
        targets_train = targets_train[:int(train_size*len(targets_train))]

        train_data = (feats_train, targets_train)
        val_data = (feats_val, targets_val)

        start_time = time.time()
        ms = train_dag(dag, train_data)
        preds = test_dag(dag, ms, val_data)
        end_time = time.time()

        scores_dict = eval_metrics(
            metrics_dict, scores_dict, preds, val_data[1])

        result = {
            m: {
                'mean': scores_dict[m][0],
                'std': 0
            } for m in scores_dict.keys()
        }

        result['time'] = end_time - start_time
        return result
    except ValueError as e:
        print('ERROR:', e)
        result = {'error': {
            'type': 'model',
            'msg': str(e)
        }}
        return result


def eval_dag_kfold(train_data, dag, metrics_dict, n_splits, train_size):
    feats, targets = train_data

    times = []
    scores_dict = {metric: [] for metric in metrics_dict.keys()}

    if n_splits < 2:
        result = {'error': {
            'type': 'splits_param',
            'msg': 'If argument "splits" is int, it must be >= 2.'
        }}
        return result

    skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)

    try:
        for train_idx, val_idx in skf.split(feats, targets):
            train_idx = train_idx[:int(train_size*len(train_idx))]

            train_data = (feats.iloc[train_idx], targets.iloc[train_idx])
            val_data = (feats.iloc[val_idx], targets.iloc[val_idx])

            start_time = time.time()
            ms = train_dag(dag, train_data)
            preds = test_dag(dag, ms, val_data)
            end_time = time.time()

            times.append(end_time - start_time)

            scores_dict = eval_metrics(
                metrics_dict, scores_dict, preds, val_data[1])

        result = {
            m: {
                'mean': np.mean(scores_dict[m]),
                'std': np.std(scores_dict[m])
            } for m in scores_dict.keys()
        }

        result['time'] = np.sum(times)

        return result
    except ValueError as e:
        result = {'error': {
            'type': 'model',
            'msg': str(e)
        }}
        return result


def eval_dag_1fold(train_data, val_data, dag, metrics_dict, train_size):
    times = []
    scores_dict = {metric: [] for metric in metrics_dict.keys()}

    train_feats, train_targets = train_data
    train_feats = train_feats[:int(train_size*len(train_feats))]
    train_targets = train_targets[:int(train_size*len(train_targets))]

    try:
        start_time = time.time()
        ms = train_dag(dag, (train_feats, train_targets))
        preds = test_dag(dag, ms, val_data)
        end_time = time.time()

        times.append(end_time - start_time)

        scores_dict = eval_metrics(
            metrics_dict, scores_dict, preds, val_data[1])
        
        result = {
            m: {
                'mean': np.mean(scores_dict[m]),
                'std': np.std(scores_dict[m])
            } for m in scores_dict.keys()
        }

        result['time'] = np.sum(times)

        return result

    except (ValueError, ZeroDivisionError) as e:
        result = {'error': {
            'type': 'model',
            'msg': str(e)
        }}
        return result


def eval_test_set(dag, train_data, test_data, metrics_dict):
    scores_dict = {metric: [] for metric in metrics_dict.keys()}

    try:
        ms = train_dag(dag, train_data)
        preds = test_dag(dag, ms, test_data)

        scores_dict = eval_metrics(
            metrics_dict, scores_dict, preds, test_data[1])

        result = {
            m: scores_dict[m][0] for m in scores_dict.keys()
        }
        return result
    except ValueError as e:
        print('ERROR:', e)
        result = {'error': {
            'type': 'model',
            'msg': str(e)
        }}
        return result


def eval_dag(dag, dataset, metrics_list, splits, test, train_size, fold):
    test_dataset = 'data/{}_{}.csv'.format(dataset, fold)

    if not os.path.isfile(test_dataset):
        result = {'error': {
            'type': 'dataset',
            'msg': 'Dataset {} does no exist.'.format(test_dataset)
        }}
        return result

    metrics_dict = {
        metric['name']: {
            'method': available_metrics[metric['metric']],
            'args': metric['args']
        } for metric in metrics_list
    }

    dag = dag_parser.normalize_dag(dag)

    if splits == 1:
        train_dataset_idx = 'data/{}_train_{}_1fold'.format(dataset, fold)
        val_dataset_idx = 'data/{}_val_{}_1fold'.format(dataset, fold)
        val_idx = fold - 1
        if val_idx == 0:
            val_idx = 5
        train_idx = [i for i in range(1, 6) if i not in [val_idx, fold]]
        if train_dataset_idx not in input_cache:
            train_df = pd.DataFrame()
            for i in train_idx:
                train_dataset_file = 'data/{}_{}.csv'.format(dataset, i)
                train_df = pd.concat(
                    [train_df, pd.read_csv(train_dataset_file, sep=';')])
            input_cache[train_dataset_idx] = train_df
        if val_dataset_idx not in input_cache:
            val_df = pd.read_csv(
                'data/{}_{}.csv'.format(dataset, val_idx), sep=';')
            input_cache[val_dataset_idx] = val_df
    else:
        train_dataset_idx = 'data/{}_train_{}'.format(dataset, fold)
        if train_dataset_idx not in input_cache:
            train_df = pd.DataFrame()
            for i in range(1, 6):
                if i == fold:
                    continue
                train_dataset_file = 'data/{}_{}.csv'.format(dataset, i)
                train_df = pd.concat(
                    [train_df, pd.read_csv(train_dataset_file, sep=';')])
            input_cache[train_dataset_idx] = train_df

    if test_dataset not in input_cache:
        input_cache[test_dataset] = pd.read_csv(test_dataset, sep=';')

    train_data = input_cache[train_dataset_idx]
    train_feats = train_data[train_data.columns[:-1]]
    train_targets = train_data[train_data.columns[-1]]

    test_data = input_cache[test_dataset]
    test_feats = test_data[test_data.columns[:-1]]
    test_targets = test_data[test_data.columns[-1]]

    le = preprocessing.LabelEncoder()

    train_ix = train_targets.index
    train_targets = pd.Series(le.fit_transform(train_targets), index=train_ix)

    test_ix = test_targets.index
    test_targets = pd.Series(le.fit_transform(test_targets), index=test_ix)

    train_data_proc = (train_feats, train_targets)
    test_data_proc = (test_feats, test_targets)

    if test:
        result = eval_test_set(dag, train_data_proc,
                               test_data_proc, metrics_dict)
    else:
        if type(splits) == int:
            if splits == 1:
                val_data = input_cache[val_dataset_idx]
                val_feats = val_data[val_data.columns[:-1]]
                val_targets = val_data[val_data.columns[-1]]

                val_ix = val_targets.index
                val_targets = pd.Series(
                    le.fit_transform(val_targets), index=val_ix)

                val_data_proc = (val_feats, val_targets)

                result = eval_dag_1fold(
                    train_data_proc, val_data_proc, dag, metrics_dict, fold)
            else:
                result = eval_dag_kfold(
                    train_data_proc, dag, metrics_dict, splits, train_size)
        elif type(splits) == float:
            result = eval_dag_train_test(
                train_data_proc, dag, metrics_dict, splits, train_size)
        else:
            result = {'error': {
                'type': 'splits_param',
                'msg': 'Parameter "splits" should '
                       'be int or float, not {}.'.format(type(splits).__name__)
            }}

    return result


def safe_dag_eval(dag, dataset, metrics_list, splits, test, train_size, fold, dag_id=None):

    try:
        return eval_dag(dag, dataset, metrics_list, splits,
                        test, train_size, fold), dag_id
    except Exception as e:
        with open('error.' + str(dag_id), 'w') as err:
            err.write(str(e) + '\n')
            for line in traceback.format_tb(e.__traceback__):
                err.write(line)
            err.write(json.dumps(dag))
    return (), dag_id
