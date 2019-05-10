import json
import time

from xmlrpc.client import ServerProxy


class RPCClient(object):
    """Opens an HTTP connection with the pipeline evaluator server.

    Arguments:
        server_url (str): URL, including port, of the server.
    """
    def __init__(self, server_url):
        self.server_url = server_url
        self.server_proxy = ServerProxy(server_url)

    def evaluate_pipeline(self, candidate, dataset, metrics_list, n_splits):
        """Evaluates a pipeline according to a list of metrics.

        Arguments:
            candidate (str):        string representing the pipeline to be evaluated.
            dataset (str):          name of the dataset to be used.
            metrics_list (list):    list of metrics used in evaluation.
            n_splits (int | float): if int, number of folds for cross-validation; 
                                    if float, proportion of the dataset to
                                    include in the test split.

        Return:
            dictionary containing mean and std of evaluated metrics, total time
            and candidate id.
        """
        _dataset = dataset + '.csv'

        cand_id = self._submit(candidate, _dataset, metrics_list, n_splits)

        return self._get_evaluated(cand_id)

    def _submit(self, candidate, dataset, metrics_list, n_splits):
        """Sends a request to the pipeline evaluator server.

        Arguments:
            candidate (str):        string representing the pipeline to be evaluated.
            dataset (str):          name of the dataset to be used.
            metrics_list (list):    list of metrics used in evaluation.
            n_splits (int | float): if int, number of folds for cross-validation; 
                                    if float, proportion of the dataset to
                                    include in the test split

        Return:
            candidate id.
        """
        return self.server_proxy.submit(
            candidate, dataset, metrics_list, n_splits)

    def _get_evaluated(self, candidate_id):
        """Retrieves the results of a pipeline evaluation from the server.

        Arguments:
            candidate_id (str): candidate id.

        Return:
            results of the pipeline evaluation or error message.
        """
        attempts = 0
        step = 2

        ev_id, results = json.loads(
            self.server_proxy.get_evaluated(candidate_id))

        while ev_id != candidate_id:
            time.sleep(attempts)
            attempts += step

            ev_id, results = json.loads(
                self.server_proxy.get_evaluated(candidate_id))

        results['id'] = ev_id

        if 'error' in results.keys():
            error_type = results['error']['type']
            error_msg = results['error']['msg']

            if error_type == 'timeout':
                return 'timeout'
            elif error_type == 'model':
                return 'invalid'
            else:
                raise Exception(error_msg)
        return results
