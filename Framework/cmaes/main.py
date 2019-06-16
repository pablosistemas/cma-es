from library.model.cmaes import Cmaes as Cmaes
from library.algorithm.run import Run as Run
from library.evaluation.felli import Felli as Felli
from library.evaluation.cigar import Cigar as Cigar
from library.evolution.generation import NextCmaes as Generation
from library.evolution.recombination import Recombination as Recombination
from library.evaluation.sklearn_evaluation import SKLearnEvaluation as SKLearnEvaluation
from library.preprocessing.get_numeric_parameter import GetNumericParameters as GetNumericParameters

import numpy as np
import traceback
import argparse
import math
import json
import sys

sys.path.append('.')

from parsers import candidate_solution
from parsers import candidate_parser as cp

from base_optimizer import BaseOptimizer


class CmaesOpt(BaseOptimizer):
    def __init__(self, server_url, dataset, metrics_list, splits, grammar_file, number_of_evaluations, seed, log_file):
        self.number_of_evaluations = number_of_evaluations

        super(CmaesOpt, self).__init__(
            server_url, dataset, metrics_list, splits, grammar_file, seed, log_file)


    def optimize(self):
        pop = candidate_solution.get_random_candidates(
            self.grammar_file, self.number_of_evaluations, self.seed, self.att)

        dictionary = dict()

        print('------------------------------------')
        print('Evaluating the pipelines....')
        print('------------------------------------')

        for pip in pop:
            try:
                sk_eval = SKLearnEvaluation(
                    pip[1], 
                    cp.load_pipeline,
                    self.evaluate_pipeline
                )
                
                initial_condition = GetNumericParameters().get(pip[1])

                model = Cmaes(Recombination().run, 30)
                model.set_initial_condition(initial_condition)

                # construtor
                algorithm = Run(model, sk_eval.run, 0.98, 'max')

                n_folds = 10
                cands_result = []
                for fold in range(1, n_folds + 1):
                    [xmin, fitness, counteval] = algorithm.run(fold)
                
                    print("#Pipeline: " + str(pip[1]))
                    print("#Evaluation performance (F1 weighted): " + str(fitness))

                    cands_result.append(fitness)

                dictionary[pip[1]] = max(cands_result)

            except Exception as e:
                traceback.format_exc()
                print("#" + str(e))

        sorted_results = list(
            sorted(dictionary.items(), key=lambda kv: kv[1]))

        # Save log
        self.write_log(sorted_results, invalid_results, opt_metric)

        return sorted_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid Search')

    # Arguments that are necessary for all optimizer.
    parser.add_argument('-d', '--dataset',
                        required=False, type=str,
                        help='Name of the dataset.',
                        default="diabetes")

    parser.add_argument('-p', '--optimizer_config',
                        type=str,
                        help='File that configures the optimizer.',
                        default="grid_search/config/optimizer.config")

    parser.add_argument('-s', '--server_config',
                        type=str,
                        help='File that configures the connection with '
                             'the server.',
                        default="grid_search/config/server.config")
    
    parser.add_argument('-g', '--grammar_file',
                        required=False, type=str,
                        help='File that contains the grammar.',
                        default="bnf/grammar_automl_v1.bnf")
    
    parser.add_argument('-seed', '--seed',
                        type=int, default=0,
                        help='Seed to control the generation of '
                             'pseudo-random numbers.')
    
    parser.add_argument('-n', '--n_vals',
                        required=False, type=int,
                        help='Number of values used in range discretization',
                        default=1)
    
    parser.add_argument('-l', '--log_file',
                        required=False, type=str,
                        help='Log filepath',
                        default="grid_search.txt")

    parser.add_argument('-t', '--execution_time',
                        required=False, type=int,
                        help='Time that the algorithm will be executed (in seconds)',
                        default=7200) #2 hours

    
    return parser.parse_args()


def main(args):
    server_url = 'http://localhost:5000'

    if args.server_config is not None:
        server_url = json.load(open(args.server_config))['serverUrl']

    metrics_list = [{
        'metric': 'f1',
        'args': {'average': 'micro'},
        'name': 'f1_score'
    }]

    splits = 5

    if args.optimizer_config is not None:
        config = json.load(open(args.optimizer_config))
        metrics_list = config['metrics']
        splits = config['splits']

    print('----- RPC Client configuration -----')
    print('Server url:', server_url)
    print('\nDataset:', args.dataset)
    print('\nMetrics:', metrics_list)
    print('\nSplits:', splits)
    print('\nLogFile:', args.log_file)

    rand_opt = CmaesOpt(server_url, args.dataset, metrics_list, splits,
                            args.grammar_file, args.n_vals, args.seed, args.log_file)

    rand_result = rand_opt.optimize()
    rs_length = len(rand_result)

    print('\n------------------------------------')
    key_best = rand_result[rs_length - 1][0]
    value_best = rand_result[rs_length - 1][1]
    print('Best pipeline: ' + str(key_best))
    print('Best pipeline\'s result (F1 weighted): ' + str(value_best))


if __name__ == "__main__":
  args = parse_args()
  
  main(args)