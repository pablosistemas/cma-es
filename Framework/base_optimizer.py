import abc

from client_server.rpc_client import RPCClient


class BaseOptimizer(abc.ABC):
    """Description @TODO

    Arguments:
        server_url (str):       url of the pipeline evaluator server.
        dataset (str):          name of the dataset to be used.
        metrics_list (list):    list of metrics used in evaluation.
        n_splits (int | float): if int, number of folds for cross-validation; 
                                if float, proportion of the dataset to
                                include in the test split.
        grammar_file (str):     file containing the grammar that generates
                                the pipelines.
        seed (int):             seed for the random number generator.

    Attributes:
        client (RPCClient):     client that communicates with the pipeline
                                evaluator server.
        dataset (str):          name of the dataset.
        metrics_list (list):    list of metrics.
        n_splits (int | float): number of folds of test size.
    """
    def __init__(self, server_url, dataset, metrics_list, n_splits, grammar_file, seed, log_file):
        self.client = RPCClient(server_url)
        self.dataset = dataset
        self.metrics_list = metrics_list
        self.n_splits = n_splits
        self.grammar_file = grammar_file
        self.seed = seed
        self.log_file = log_file
        self.att = self.client.get_num_feats_dataset(dataset)

        with open(log_file, 'w') as f:
            f.write('')

    def evaluate_pipeline(self, candidate, fold, test=False, train_size=1):
        """Evaluates a pipeline.

        Arguments:
            candidate (str):        string representing the pipeline to be evaluated.
            fold (int):             cross-validation fold to use.
            test (bool):            evaluate pipeline on the test set.
            train_size (float):     proportion of the training fold to actually
                                    use to train the model.

        Return:
            dictionary containing mean and std of evaluated metrics, total time
            and candidate id. If test=True, dictionary contains a single value
            for each metric.
        """
        results = self.client.evaluate_pipeline(
            candidate, self.dataset, self.metrics_list, self.n_splits,
            test, train_size, fold)

        return results

    def write_log(self, results, invalid_results, opt_metric):
        """Creates a log file with the results.
        """
        with open(self.log_file, 'a') as f:
            res = results[0]
            f.write(res[1]['timestamp'] + ';')
            f.write(res[0] + ';')
            f.write(str(res[1][opt_metric]['mean']) + ';')
            f.write('B\n')

            # Best and valid results
            for res in results[1:]:
                f.write(res[1]['timestamp'] + ';')
                f.write(res[0] + ';')
                f.write(str(res[1][opt_metric]['mean']) + ';')
                f.write('V\n')

            # Invalid and timeout
            for res in invalid_results:
                f.write(invalid_results[res]['timestamp'] + ';')
                f.write(res + ';;')
                if invalid_results[res]['result'] == 'invalid':
                    f.write('I\n')
                else:
                    f.write('T\n')

    @abc.abstractmethod
    def optimize(self):
        """Find the best machine learning pipeline for a dataset according
        to a set of metrics.

        Return:
            list of best pipelines for each cross-validation fold. Each
            pipeline is represented by a 3-tuple: tree and string-based
            representation and dictionary with the values of the evaluated
            metrics;
            list of metrics for the best pipeline of each fold evaluated on
            the test set.
        """
        pass
