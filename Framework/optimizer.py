import argparse
import json

from base_optimizer import BaseOptimizer


class MyOpt(BaseOptimizer):
    def __init__(self, server_url, dataset, metrics_list, splits):
        super(MyOpt, self).__init__(server_url, dataset, metrics_list, splits)

    def optimize(self):
        # candidate = '{"input": [[], "input", ["1:0"]], ' \
        #             '"1": [["1:0"], ["gaussianNB", {}], []]}'

        candidate_obj = {
            "input": [[], "input", ["IN:0"]],
            "PCA": [["IN:0"], ["PCA", {"n_components": 5}], ["PCA:0"]],
            "MLP": [["PCA:0"], ["MLP", {"max_iter": 20, "verbose": "True"}], ["MLP:0"]],
            "DT1": [["PCA:0"], ["DT", {}], ["DT1:0"]],
            "vote1": [["MLP:0", "DT1:0"], ["vote", {}], []]
        }
        candidate = json.dumps(candidate_obj)
        # print('candidate', candidate)

        results = self.evaluate_pipeline(candidate)

        print('Results:')
        print(json.dumps(results, indent=2, sort_keys=True))

        return candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description='TODO')

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

    return parser.parse_args()


def main(args):
    server_url = 'http://automl.speed.dcc.ufmg.br:80'

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
    print('\n------------------------------------')

    opt = MyOpt(server_url, args.dataset, metrics_list, splits)
    best_pipeline = opt.optimize()

    print('\n------------------------------------')
    print('Best pipeline:')
    print(best_pipeline)


if __name__ == '__main__':
    args = parse_args()

    main(args)
