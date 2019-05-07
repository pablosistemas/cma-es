from library.model.cmaes import Cmaes as Cmaes
from library.algorithm.run import Run as Run
from library.evaluation.felli import Felli as Felli
from library.evaluation.cigar import Cigar as Cigar
from library.evolution.generation import NextCmaes as Generation
from library.evolution.recombination import Recombination as Recombination
from library.evaluation.sklearn_evaluation import SKLearnEvaluation as SKLearnEvaluation
from library.preprocessing.get_numeric_parameter import GetNumericParameters as GetNumericParameters

import numpy as np
import argparse
import math
import json
import sys

sys.path.append('.')

from parsers import candidate_solution
from parsers import candidate_parser as cp

from base_optimizer import BaseOptimizer


class CmaesOpt(BaseOptimizer):
    def __init__(self, server_url, dataset, metrics_list, splits, grammar_file, number_of_evaluations, seed):
        self.number_of_evaluations = number_of_evaluations

        super(CmaesOpt, self).__init__(
            server_url, dataset, metrics_list, splits, grammar_file, seed)


    def optimize(self):
        pop = candidate_solution.get_random_candidates(
            self.grammar_file, self.number_of_evaluations, self.seed)

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

                algorithm = Run(model, sk_eval.run, 0.98, 'max')
                [xmin, fitness, counteval] = algorithm.run()
                
                print("#Pipeline: " + str(pip[1]))
                print("#Pipeline's parse tree: " + str(pip[2]))
                print("#Evaluation performance (F1 weighted): " + str(fitness))

                dictionary[pip[1]] = fitness

            except Exception as e:
                print("#" + str(e))

        sorted_dictionary = list(
            sorted(dictionary.items(), key=lambda kv: kv[1]))

        return sorted_dictionary


def parse_args():
    parser = argparse.ArgumentParser(
        description='TODO')

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

    # This argument is specific for this optimizer.
    # You can define your own stopping criterion.
    parser.add_argument('-n', '--number_of_evaluations',
                        type=int, default=5,
                        help='Number of pipeline evaluations considered in '
                             'the random search.')

    return parser.parse_args()


def main2(args):
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

    rand_opt = CmaesOpt(server_url, args.dataset, metrics_list, splits,
                            args.grammar_file, args.number_of_evaluations, args.seed)

    rand_result = rand_opt.optimize()
    rs_length = len(rand_result)

    print('\n------------------------------------')
    key_best = rand_result[rs_length - 1][0]
    value_best = rand_result[rs_length - 1][1]
    print('Best pipeline: ' + str(key_best))
    print('Best pipeline\'s result (F1 weighted): ' + str(value_best))


def main():

  dimension_size = 10
  model = Cmaes(dimension_size, Recombination().run)
  algorithm = Run(model, Cigar().run)
  [xmin, fitness, counteval] = algorithm.run()

  print("%s : %s" %(str(counteval), str(fitness)))
  print(xmin)

  return

if __name__ == "__main__":
  args = parse_args()
#   main()
  main2(args)