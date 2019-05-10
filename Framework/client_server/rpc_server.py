import argparse
import json
import multiprocessing
import time

from hashlib import md5
from xmlrpc.server import SimpleXMLRPCServer

import dag_evaluator

import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

stop_server = False


def exec_timeout(func, args, timeout):
    pool = multiprocessing.Pool(1, maxtasksperchild=1)
    result = pool.apply_async(func, args)
    pool.close()

    try:
        ind_scores, ind_id = result.get(timeout)
        return ind_scores, ind_id
    except multiprocessing.TimeoutError:
        pool.terminate()
        print('\nTIME LIMIT EXCEEDED\n')
        result = {'error': {
            'type': 'timeout',
            'msg': 'Time limit exceeded.'
        }}
        return result, args[4]


def eval_dags(inputs, evaluated_list, timeout):
    while True:
        try:
            ind_id, ind_dag, filename, metrics_list, splits = inputs.get(
                block=False)

            ind_scores, _ind_id = exec_timeout(
                func=dag_evaluator.safe_dag_eval,
                args=[ind_dag, filename, metrics_list, splits, ind_id],
                timeout=timeout
            )

            assert ind_id == _ind_id

            evaluated_list.append([ind_id, ind_scores])
        except Exception:
            time.sleep(1)


class DagEvalServer:

    def __init__(self, n_cpus, timeout):
        self.manager = multiprocessing.Manager()
        self.evaluated_list = self.manager.list()

        self.n_cpus = n_cpus if n_cpus >= 0 else multiprocessing.cpu_count()
        self.timeout = timeout

        self.gen_number = 0

        self.inputs = multiprocessing.Queue()

        self.processes = [multiprocessing.Process(target=eval_dags, args=(
            self.inputs,
            self.evaluated_list,
            self.timeout)
        ) for _ in range(n_cpus)]

        for p in self.processes:
            p.start()

    def submit(self, candidate_string, datafile, metrics_list, splits):
        candidate = json.loads(candidate_string)

        sub_time = time.time()

        m = md5()
        m.update((candidate_string + str(sub_time)).encode())
        cand_id = m.hexdigest()

        self.inputs.put((cand_id, candidate, datafile, metrics_list, splits))

        return cand_id

    def get_evaluated(self, ind_id):
        for ind in self.evaluated_list:
            if ind[0] == ind_id:
                self.evaluated_list.remove(ind)
                return json.dumps(ind)

        return json.dumps([None, None])

    def get_core_count(self):
        return json.dumps(self.n_cpus)

    def kill_workers(self):
        for p in self.processes:
            p.terminate()

    def quit(self):
        global stop_server
        stop_server = True
        return json.dumps('OK')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Start server that evaluates machine learning pipelines.')

    parser.add_argument('-p', '--port-number',
                        required=True, type=int, default=80,
                        help='Port in which the server is supposed to run.')

    parser.add_argument('-n', '--n_cpus',
                        type=int, default=1,
                        help='Number of CPUs to use.')

    parser.add_argument('-t', '--timeout',
                        type=int, default=300,
                        help='Maximum execution time (in seconds) '
                        'for a single pipeline.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    n_cpus = args.n_cpus
    timeout = args.timeout

    port_number = args.port_number
    server_url = '0.0.0.0'

    # if args.config:
    #     config = json.load(open(args.config))
    #     server_url = config['serverUrl']
    #     port_number = int(server_url.split(':')[-1])
    #     timeout = int(config['timeout'])

    print('=============== Server Settings:')
    print('Server URL: ', server_url)
    print('Port Number:', port_number)
    print('Num CPUs:', n_cpus)
    print('Timeout: {}s'.format(timeout))
    print()

    eval_server = DagEvalServer(n_cpus, timeout)

    server = SimpleXMLRPCServer((server_url, port_number))
    server.register_instance(eval_server)

    try:
        server.serve_forever()
    except Exception as e:
        stop_server = True
        server.kill_workers()
        print("ERROR: ", str(e))
        print(repr(e))
