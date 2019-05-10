import abc

from client_server.rpc_client import RPCClient


class BaseOptimizer(abc.ABC):
    """Description @TODO

    Arguments:
        candidate (str):        string representing the pipeline to be evaluated.
        dataset (str):          name of the dataset to be used.
        metrics_list (list):    list of metrics used in evaluation.
        n_splits (int | float): if int, number of folds for cross-validation; 
                                if float, proportion of the dataset to
                                include in the test split.

    Attributes:
        client (RPCClient):     client that communicates with the pipeline
                                evaluator server.
        dataset (str):          name of the dataset.
        metrics_list (list):    list of metrics.
        n_splits (int | float): number of folds of test size.
    """
    def __init__(self, server_url, dataset, metrics_list, n_splits, grammar_file, seed):
        self.client = RPCClient(server_url)
        self.dataset = dataset
        self.metrics_list = metrics_list
        self.n_splits = n_splits
        self.grammar_file = grammar_file
        self.seed = seed

    def evaluate_pipeline(self, candidate):
        """Evaluates a pipeline.

        Arguments:
            candidate (str): string representing the pipeline to be evaluated.

        Return:
            dictionary containing mean and std of evaluated metrics, total time
            and candidate id.
        """
        results = self.client.evaluate_pipeline(
            candidate, self.dataset, self.metrics_list, self.n_splits)

        return results

    @abc.abstractmethod
    def optimize(self):
        """Find the best machine learning pipeline for a dataset according
        to a set of metrics.

        Return:
            tree-based representation of the pipeline and a dictionary with
            the values of the evaluated metrics.
        """
        pass
