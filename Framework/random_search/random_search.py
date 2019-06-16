import argparse
import json
import sys
import numpy as np
import time

sys.path.append('.')

from parsers import candidate_solution
from parsers import candidate_parser as cp
from base_optimizer import BaseOptimizer

verbose = True


class RandomOpt(BaseOptimizer):
    def __init__(self, server_url, dataset, metrics_list, splits, grammar_file, number_of_evaluations, seed, log_file):
        self.number_of_evaluations = number_of_evaluations
        super(RandomOpt, self).__init__(server_url, dataset, metrics_list,
                                        splits, grammar_file, seed, log_file)

    def optimize(self, verbose=False):
        """
        For each cross validation fold, generates n random candidates and
        chooses the best of them accordind to an evaluation metric.

        Return:
            list of best candidate for each fold;
            list of evaluation metrics for the best candidate for each fold 
            on the test set.
        """
        # Define metric to be optimized
        opt_metric = 'f1_weighted'

        results_train = []
        results_test = []

        # Cross-validation
        n_folds = 5
        for fold in range(1, n_folds + 1):
            print('Evaluating fold ' + str(fold), '\n')

            # Get population of random candidates
            population = candidate_solution.get_random_candidates(
                self.grammar_file, self.number_of_evaluations,
                self.seed, self.att)

            results = {}
            invalid_results = {}

            for i, pipeline in enumerate(population):
                cand_tree = pipeline[0]
                cand_string = pipeline[1]

                # Get representation understood by the server
                try:
                    candidate = cp.load_pipeline(cand_string)
                except NotImplementedError as e:
                    print(e)
                    continue

                if verbose:
                    print('Candidate {}: {}'.format(i + 1, cand_string))

                # Evaluate candidate
                cand_result = self.evaluate_pipeline(candidate, fold)

                # Create results dictionary
                if cand_result not in ['invalid', 'timeout']:
                    results[cand_string] = cand_result
                    results[cand_string]['tree'] = cand_tree
                    results[cand_string]['timestamp'] = time.ctime()
                    if verbose:
                        print(opt_metric, ':', cand_result[opt_metric], '\n')
                else:
                    invalid_results[cand_string] = {'result': cand_result}
                    invalid_results[cand_string]['tree'] = cand_tree
                    invalid_results[cand_string]['timestamp'] = time.ctime()
                    if verbose:
                        print(cand_result, '\n')

            if len(results) == 0:
                continue

            # Sort the results according to opt_metric
            sorted_results = sorted(
                results.items(), reverse=True,
                key=lambda cand: cand[1][opt_metric]['mean'])

            # Save log
            self.write_log(sorted_results, invalid_results, opt_metric)

            # Get best candidate
            best_candidate = sorted_results[0]
            best_result = best_candidate[1]
            best_tree = best_result['tree']
            best_string = best_candidate[0]

            del best_result['tree']
            del best_result['id']
            del best_result['timestamp']

            # Evaluate best candidate on the test set
            best_cand_server = cp.load_pipeline(best_string)
            test_result = self.evaluate_pipeline(
                best_cand_server, fold, test=True)

            del test_result['id']

            # Save results
            results_train.append(best_candidate)
            results_test.append(test_result)

            print('------------------------------------\n')

        return results_train, results_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run an optimizer that randomly creates candidate '
                    'pipelines based on a grammar.')

    # Arguments that are necessary for all optimizer.
    parser.add_argument('-d', '--dataset',
                        required=True, type=str,
                        help='Name of the dataset.')

    parser.add_argument('-p', '--optimizer_config',
                        type=str,
                        help='File that configures the optimizer.')

    parser.add_argument('-s', '--server_config',
                        type=str,
                        help='File that configures the connection with '
                             'the server.')
    parser.add_argument('-g', '--grammar_file',
                        required=True, type=str,
                        help='File that contains the grammar.')

    parser.add_argument('-seed', '--seed',
                        type=int, default=0,
                        help='Seed to control the generation of '
                             'pseudo-random numbers.')

    parser.add_argument('-l', '--log_file',
                        required=True, type=str,
                        help='File where log will be written.')

    # This argument is specific for this optimizer.
    # You can define your own stopping criterion.
    parser.add_argument('-n', '--number_of_evaluations',
                        type=int, default=5,
                        help='Number of pipeline evaluations considered in '
                             'the random search.')

    return parser.parse_args()


def print_client_config(url, dataset, metrics, splits):
    print('----- RPC Client configuration -----')
    print('Server url:', url)
    print('\nDataset:', dataset)
    print('\nMetrics:', metrics)
    if isinstance(splits, int):
        print('\nFolds:', splits)
    if isinstance(splits, float):
        print('\nValidation size:', splits)
    print('------------------------------------\n')


def get_means_train_test(results_train, results_test):
    """Calculates the mean of the evaluation metrics for train and test sets.

    Arguments:
        results_train (list of dict): training results.
        results_test (list of dict):  test results.

    Returns:
        mean and std of the metrics for the training and test results.
    """
    train_metrics = {m: [] for m in results_train[0][1]}
    for result in results_train:
        metrics = result[1]
        for metric in metrics.keys():
            if metric == 'time':
                train_metrics['time'].append(metrics[metric])
            else:
                train_metrics[metric].append(metrics[metric]['mean'])
    train_metrics_ = {}
    for metric in train_metrics:
        train_metrics_[metric] = {}
        train_metrics_[metric]['mean'] = np.mean(train_metrics[metric])
        train_metrics_[metric]['std'] = np.std(train_metrics[metric])

    test_metrics = {m: [] for m in results_test[0]}
    for result in results_test:
        for metric in result:
            test_metrics[metric].append(result[metric])
    test_metrics_ = {}
    for metric in test_metrics:
        test_metrics_[metric] = {}
        test_metrics_[metric]['mean'] = np.mean(test_metrics[metric])
        test_metrics_[metric]['std'] = np.std(test_metrics[metric])

    return train_metrics_, test_metrics_


def main(args):
    # Default parameters
    server_url = 'http://automl.speed.dcc.ufmg.br:80'
    metrics_list = [{
        'metric': 'f1',
        'args': {'average': 'micro'},
        'name': 'f1_score'
    }]
    splits = 5

    # Read parameters from arguments
    if args.server_config is not None:
        server_url = json.load(open(args.server_config))['serverUrl']

    if args.optimizer_config is not None:
        config = json.load(open(args.optimizer_config))
        metrics_list = config['metrics']
        splits = config['splits']

    # Print client configuration information
    print_client_config(server_url, args.dataset, metrics_list, splits)

    # Run random search
    rand_opt = RandomOpt(server_url, args.dataset, metrics_list, splits,
                         args.grammar_file, args.number_of_evaluations,
                         args.seed, args.log_file)
    results_train, results_test = rand_opt.optimize(verbose=verbose)

    train_means, test_means = get_means_train_test(results_train, results_test)

    print('Training set results:\n')
    print(json.dumps(train_means, indent=2), '\n\n')

    print('Test set results:\n')
    print(json.dumps(test_means, indent=2))


if __name__ == '__main__':
    args = parse_args()

    main(args)
