# AutoML Framework

The AutoML Framework is composed of 4 main components: a pipeline evaluator server, a pipeline evaluator client, a grammar parser, and an optimizer. The students must implement an optimizer using the parser to generate the candidate pipelines.

## Dependencies

The dependencies of the AutoML Framework are:

* python 3.x
* pip3

There is a set of other dependencies listed in the file ***requirements.txt***, which can be installed running `pip3 install -r requirements.txt`.

## Pipeline evaluator server

The pipeline evaluator server is implemented in ***client_server/rpc_server.py***. There is a version running at `http://automl.speed.dcc.ufmg.br:80`.

The server receives an HTTP request containing a string that represents the pipeline to be evaluated, the name of the dataset to be used for evaluation (see the [dataset](#datasets) section for a list of available datasets), a list of evaluation metrics and a number corresponding to the number of folds for cross-validation or the proportion of the dataset to be used as the test set. It then  executes the pipeline and returns the values of the evaluated metrics.

The Python class RPCClient (file ***client_server/rpc_client.py***) provides a direct interface with the server that can be used by the optimizer. Should you decide to implement the optimizer in a programming language other than Python 3, you must reimplement this interface.

### Server execution <a name="server_exec"></a>

The server can also be run locally (which is recommended during the implementation of the optimizer). To do so, run the following command line:

```python3 client_server/rpc_server.py -p <port_num> -n <num_cpus>  -t <evaluation_timeout>```

The arguments `<port_num>` and `<num_cpus>` correspond to the port number and number of CPUs used by the server during evaluation. The argument  `<evaluation_timeout>` establishes a timeout for pipeline evaluation.

## Grammar parser

The optimizer’s search space is defined by a grammar. A grammar **G** is represented by a 4-tuple &lt;N, T, P, S&gt;, where  **N** represents a set of non-terminals,  **T** a set of terminals, **P** a set of production rules and  **S** (a member of  **N**) the start symbol. We will use the Backus Naur Form (BNF) to represent grammars, for instance:

```bnf
<start> ::= <preprocessing> <algorithm> | <algorithm>
<preprocessing> ::= <kBest> | ...
<kBest> ::= kBest <k>
<k> ::= RANDATT(1,ATT-1)
<algorithm> ::= gaussianNB | <MLP> | ...
<MLP> ::= MLP <learning_rate> <learning_rate_init> <momentum> <max_iter> <activation>
<learning_rate> ::= constant | invscaling | adaptive
<learning_rate_init> ::= RANDFLOAT(0.1,1.0)
<momentum> ::= RANDFLOAT(0.0,1.0)
<max_iter> ::= RANDINT(10,10000)
<activation> ::= identity | logistic | tanh | relu
```

Symbols wrapped in "**<>**" represent non-terminals, whereas terminals are not bounded by "**<>**". The special symbol "**|**"  represents  a choice. The choice of one among all production rules separated by "**|**" is made using a uniform probability distribution (i.e., all elements are equally likely to occur in an individual).

"RANDINT(n,m)" means that the framework will specify an integer number between n and m for the given terminal. "RANDATT(n,ATT-1)", in turn, indicates that an integer number between n and the number of attributes minus one of the dataset will be chosen. This is important for preprocessing methods. Finally, "RANDFLOAT(p, q)" defines a real number between p and q.

The BNF parser can be found in ***parsers/grammar_parser.py***. This python file reads a grammar from a BNF file, structuring the productions, non-terminals and terminals. The first version of the AutoML grammar can be found in ***bnf/grammar_automl_v0.bnf***. The second version of the AutoML grammar can be found in ***bnf/grammar_automl_v1.bnf***

Given the parsed grammar, you need to use it to generate a (set of) candidate pipeline(s). In this framework, we use a tree-based representation, derived from the expansion of the production rules of the grammar. To create a candidate pipeline, a mapping process is performed taking the terminals from the tree (located at the leaf nodes) and constructing a valid pipeline from them. The python file  ***parsers/candidate_solution.py*** does so by using the following methods:

* **legal_productions()**: Returns the available production rule choices (from the grammar) for a node given a specific depth limit (determining the size of the produced candidate pipelines). For now, we set this depth limit to 100, which is equivalent to disregarding a maximum depth. This means that any combination of machine learning algorithms and hyperparameters found in the grammar will be considered by the proposed optimizer at any search point. This procedure has a parameter "method" that defines how the trees are generated. The "full" method expands every branch of the tree until the depth limit. On the other hand, the “random” method does not take the depth of the tree into consideration (unless it exceeds the limit). By default, this framework only uses the random method to generate candidate pipelines.  For now, we recommend that you do not change the values of both the depth and the initialization method.

* **get_random_cand()**: Returns a random candidate pipeline by randomly choosing the production rules of the grammar. The output of the method is a pair &lt;T, S&gt;, where **T** is a tree-based representation of the pipeline and **S** is a string-based representation which comes from the leaves of the tree (i.e. the grammar terminals).

* **get_random_candidates()**: Returns a set of candidate pipelines by making successive calls to **get_random_cand()**. The method has a parameter that defines the number of pipelines to be returned.

Basically, the tree-based representation of a pipeline defines the grammar derivations used to reach the grammar terminals, which are represented by the the leaf nodes. For instance, if we have *Select K Best* as the preprocessing method with K equals to 3, and a *Gaussian Naïve Bayes* as the classification algorithm, we would have the following representation of the tree for this pipeline:

```string
[
    [0, '<Start>', '<preprocessing>', None],
    [1, '<preprocessing>', '<kBest>', '<Start>'],
    [2, '<kBest>', 'kBest ', '<preprocessing>'],
    [3, '<k>', '3', '<kBest>'],
    [1, '<algorithm>', 'gaussianNB', '<Start>']
]
```

The tree is represented by an array of arrays, where the outer array represents the grammar derivations and the inner arrays have the instantiations of these productions, composed of the current tree depth, the production rule, the terminal/non-terminal chosen for that production rule, and the parent node/production. With this tree, we can generate a string representing the "raw" pipeline by looking at its leaves. In the example above, we would have the following string:

> kBest 3 gaussianNB

Now we need to translate the grammar-based string to what is understood by the (scikit-learn) evaluator. This evaluator also receives as input a string, but in a format that represents a directed acyclic graph (DAG). For example, considering that the the expansion of the production rules of the grammar (found above) generated the previous string, the candidate parser (found at ***parsers/candidate_parser.py***) will read this string and produce the following:

```string
'{
    "input": [[], "input", ["IN:0"]], 
    "kBest_1": [["IN:0"], ["kBest", {"k":3}], ["PRE:0"]],
    "gaussianNB_1": [["PRE:0"], ["gaussianNB", {}], []]
}'
```

Each line in the format above has a label, followed by an input, a processing operation and associated hyperparameters (represented by a dictionary) and an output. In the case above,  the pipeline receives as input a dataset and returns the read dataset using the tag "IN:0" to indicate it. Next, this dataset is set as input to the *Select K Best* method (represented by the tag "kBest_1"), and returns the preprocessed dataset, using the tag "PRE:0” to indicate it. Note that kBest has a hyperparameter k, listed together with the method as a dictionary ({"k":3}).

Finally, the preprocessor output is set as an input for Gaussian Naïve Bayes classifier (tag "gaussianNB_1"), which returns the predictions for that input dataset.

## Optimizer

Your optimizer should extend the BaseOptimizer class and implement the** optimize()** method. This base class contains an instance of the pipeline evaluator client, so you don’t have to worry about the communication with the server.

A random optimizer was provided as an example. You should use the file ***random_search/random_search.py*** as a template and implement your own **optimize()** method. This method will output the results of the best pipeline found as a 3-tuple: the first element is the tree representation of the pipeline; the second is its string representation; the third is the dictionary containing the values of the metrics returned by **evaluate_pipeline()**. The best pipeline should then be evaluated on the test set by calling **evaluate_pipeline()** with the optional argument `test` set to `True`.

All your code, including the configuration files (see below), must be inside a directory with the name of your optimizer. **You are not allowed to modify other files.**

### evaluate_pipeline() method

This method takes a string representing a candidate pipeline, sends a request to the server and returns the pipeline’s evaluation according to the metrics passed to the optimizer. The result is a dictionary containing the mean and standard deviation of the evaluated metrics, total execution time and the candidate solution id. **You should save this id.** If an error occurs during the evaluation of the pipeline, the server generates an error log, which is identified by the pipeline id.

Three types of errors can occur during evaluation:

* Timeout: The server has a time budget to evaluate each candidate pipeline. If this limit is exceeded, **evaluate_pipeline()** returns the string "timeout".

* Invalid pipeline: If the pipeline is invalid (e.g. calling a method with parameters that are incompatible with the data), **evaluate_pipeline()** returns the string "invalid".

* Optimizer configuration error: If there is an error in the optimizer configuration file (e.g. parameter "splits" &lt; 2), **evaluate_pipeline()** throws an exception, which terminates the optimizer execution.

If this method is called with the optional argument `test` set to `True`, the server trains the pipeline with the complete training set (without spliting it into training and validation sets) and evaluates the model on the test set. In this case, the returned dictionary contains a single value for each metric.

The method also receives an argument that chooses the cross-validation fold to use (see section [optimizer evaluation](#opt-eval)).

### Optimizer execution

To execute the optimizer, you should run the following command line (using random search as an example) in the root directory of the project :

```bash
python3 random_search/random_search.py -d <dataset_name> -s random_search/config/server.config
-p random_search/config/optimizer.config -g bnf/grammar_automl_v1.bnf -seed <seed> -l <log_file> [-n <num_evaluations]
```

These parameters define the name of the dataset being evaluated (`<dataset_name>`), the random generator seed, which should be stored to allow experiment reproducibility (`<seed>`), the file where to log will be written (`<log_file>`) and an optional number of candidate solution evaluations (useful for iterative methods such as Bayesian optimization).  The ***server.config*** and ***optimizer.config*** files are defined below.

### Optimizer evaluation <a name="opt-eval"></a>

The optimizer will be evaluated using 10-fold cross-validation. The datasets (see the [dataset](#datasets) section) were split into 10 folds. For each fold, 90% of the dataset is used for training, whereas 10% is used for testing. The current fold being used is defined by an argument passed to the  **evaluate_pipeline()** method. The file ***random_search/random_search.py*** shows an example of how to perform cross-validation.

> Note that this is not the same as the cross-validation used during the training of the models.

### Server configuration file

The server configuration file contains a JSON dictionary with a single key-value pair representing the URL of the pipeline evaluator server. An example of a server configuration file is given below:

```json
{
    "serverUrl": "http://automl.speed.dcc.ufmg.br:80"
}
```

You can use this file to easily change from your local server and the remote one provided for the AutoML course. To access a local version of the server, simply replace the URL with `"http://localhost:<port_num>"`, where `<port_num>` is the port number defined in the [server execution](#server_exec).

### Optimizer configuration file

This file defines the list of metrics to be evaluated by the server and the train-test split strategy. An example of an optimizer configuration file is given below:

```json
{
    "metrics": [
        {"metric": "accuracy", "args": {}, "name": "accuracy"},
        {"metric": "f1", "args": {"average": "micro"}, "name": "f1_micro"},
        {"metric": "f1", "args": {"average": "macro"}, "name": "f1_macro"}
    ],
    "splits": 5
}
```

The "metrics" field contains a list of metrics to be evaluated. Each item is a dictionary. The "metric" field is the name of the metric (see the [next session](#metrics) for available metrics and the reference to their documentation); the "args" field is a dictionary containing the parameters supported by the method (see documentation); the "name" field is used to give a unique name to each configuration of the metrics.

The "splits" field defines how the dataset will be divided into training and test sets. If it is an integer (&geq; 2), it corresponds to the number of cross-validation folds. If it is a real number (&gt; 0 and &lt; 1), it corresponds to the proportion of the dataset that will be used as test set.

### Log file

The base optimizer class implements a method that generates a log of the optimization process. The log contains the following information:

* Time stamp
* Solution string as defined by the grammar
* Value of weighted f1-score
* A character indicating the status of the solution:
  * B: best solution of the current iteration
  * V: a valid solution (but not the best)
  * I: an invalid solution (because one constraint was not accounted by the grammar)
  * T: a solution that received a timeout during the pipeline execution


## Available metrics <a name="metrics"></a>

* accuracy (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)>

* f1 (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)>

* recall (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)>

* precision (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)>

## Datasets <a name="datasets"></a>

Currently, there are 4 datasets available, which are stored on the server:

|        Dataset       | # instances | # attributes | # classes |
|:--------------------:|:-----------:|:------------:|:---------:|
|     [ml-prove][1]    |     6118    |      51      |     6     |
|       [wilt][2]      |     4839    |       5      |     2     |
| [statlog-segment][3] |     2310    |      19      |     7     |
|     [diabetes][4]    |     768     |       8      |     2     |

[1]: https://archive.ics.uci.edu/ml/datasets/First-order+theorem+proving
[2]: http://archive.ics.uci.edu/ml/datasets/wilt
[3]: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Image+Segmentation%29
[4]: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Support

Any errors should be reported via Moodle or by opening an issue on the repository. Please include the id of the pipeline that caused the error, if applicable.

---

<sub>Last update: Thursday, 09 May 2019 10:05 GMT-03:00</sub>
