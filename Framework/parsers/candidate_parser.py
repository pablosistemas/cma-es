def load_pipeline(pipeline_string):
    """Load the pipeline from the candidate's solution (according to the
    string generated by the GP grammar) for DAG evaluate representation.

    Parameters
    ----------

    pipeline : string
       A string contaning the method and the respective arguments.

    """
    # algorithms in the pipeline:
    algs = pipeline_string.strip().split()

    # Setting the choices of the algorithms and preprocessors:
    prep_choices = ['SimpleImputer', 'Normalizer', 'MinMaxScaler',
                    'MaxAbsScaler', 'RobustScaler', 'StandardScaler',
                    'VarianceThreshold', 'SelectKBest', 'PCA',
                    'IncrementalPCA', 'FastICA', 'GaussianRandomProjection',
                    'SparseRandomProjection', 'FeatureAgglomeration',
                    'RBFSampler', 'Nystroem', 'TruncatedSVD', 'Binarizer']
    alg_choices = ['gaussianNB', 'BernoulliNB', 'MultinomialNB', 'SVC', 'MLP',
                   'ComplementNB', 'NuSVC', 'LogisticRegression', 'Perceptron',
                   'SGD', 'LDA', 'QDA', 'KNearestNeighbors', 'RadiusNeighbors',
                   'Centroid', 'Ridge', 'RidgeCCV',  'ExtraTree', 'DT',
                   'RandomForest', 'ExtraTrees', 'AdaBoost', 'GradientBoosting']

    # Inherent part of the translation:
    input_part = '{"input": [[], "input", ["IN:0"]], '

    i = 0
    class_part = ''
    prep_part = ''
    prep_in_out = '\"IN:0\"'

    while (i < len(algs)):
        if(algs[i] in prep_choices):
            try:
                prep_alg = load_preprocessing_alg(algs, i, prep_in_out)
                prep_part += prep_alg[0]
                i += prep_alg[1]
                prep_in_out = prep_alg[2]
            except NotImplementedError as e:
                raise e

        elif(algs[i] in alg_choices):
            try:
                if(i != 0):
                    class_alg = load_classification_alg(algs, prep_in_out, i)
                else:
                    class_alg = load_classification_alg(algs, '\"IN:0\"', i)
                class_part = class_alg[0]
                i += class_alg[1]
            except NotImplementedError as e:
                raise e
        else:
            print('Error during parsing!', algs[i], '\n')
            break

    # Divide the translation into input, preprocessing and classification:
    output = input_part + prep_part + class_part

    return output


def load_preprocessing_alg(alg, i, prep_input):
    if alg[i] in preproc_methods and preproc_methods[alg[i]] is not None:
        return preproc_methods[alg[i]](alg, i, prep_input)
    else:
        raise NotImplementedError('Preprocessing method ' + alg[i] +
                                  ' not implemented.')


def load_classification_alg(alg, input_param, i):
    if alg[i] in classif_methods and classif_methods[alg[i]] is not None:
        return classif_methods[alg[i]](alg, input_param, i)
    else:
        raise NotImplementedError('Classification method ' + alg[i] +
                                  ' not implemented.')


# Preprocessing methods

def PCA(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"PCA\": [[' + prep_input + '], [\"PCA\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"whiten\":\"' + alg[i + 2] + '\", '
    dag_output += '\"svd_solver\":\"' + alg[i + 3] + '\", '
    dag_output += '\"tol\":' + alg[i + 4] + ', '
    if(alg[i + 5] == "auto"):
        dag_output += '\"iterated_power\":\"' + \
            alg[i + 5] + '\"}], [' + prep_output + ']], '
    else:
        dag_output += '\"iterated_power\":' + \
            alg[i + 5] + '}], [' + prep_output + ']], '
    return dag_output, 6, prep_output


def kBest(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"SelectKBest\": [[' + prep_input + '], [\"SelectKBest\", '
    dag_output += '{\"k\":' + alg[i + 1] + '}], [' + prep_output + ']], '

    return dag_output, 2, prep_output


def simp(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"SimpleImputer\": [[' + \
        prep_input + '], [\"SimpleImputer\", '
    dag_output += '{\"strategy\":\"' + \
        alg[i + 1] + '\"}], [' + prep_output + ']], '
    return dag_output, 2, prep_output


def norm(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"Normalizer\": [[' + prep_input + '], [\"Normalizer\", '
    dag_output += '{\"norm\":\"' + \
        alg[i + 1] + '\"}], [' + prep_output + ']], '
    return dag_output, 2, prep_output


def minmax(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"MinMaxScaler\": [[' + prep_input + '], [\"MinMaxScaler\", '
    dag_output += '{}], [' + prep_output + ']], '
    return dag_output, 1, prep_output


def maxabs(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"MaxAbsScaler\": [[' + prep_input + '], [\"MaxAbsScaler\", '
    dag_output += '{}], [' + prep_output + ']], '
    return dag_output, 1, prep_output


def robust(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"RobustScaler\": [[' + prep_input + '], [\"RobustScaler\", '
    dag_output += '{\"with_scaling\":\"' + alg[i + 1] + '\", '
    dag_output += '\"with_centering\":\"' + \
        alg[i + 2] + '\"}], [' + prep_output + ']], '
    return dag_output, 3, prep_output


def standard(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"StandardScaler\": [[' + \
        prep_input + '], [\"StandardScaler\", '
    dag_output += '{\"with_std\":\"' + alg[i + 1] + '\", '
    dag_output += '\"with_mean\":\"' + \
        alg[i + 2] + '\"}], [' + prep_output + ']], '
    return dag_output, 3, prep_output


def variance(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"VarianceThreshold\": [[' + \
        prep_input + '], [\"VarianceThreshold\", '
    dag_output += '{}], [' + prep_output + ']], '
    return dag_output, 1, prep_output


def incrementalPCA(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"IncrementalPCA\": [[' + \
        prep_input + '], [\"IncrementalPCA\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"whiten\":\"' + \
        alg[i + 2] + '\"}], [' + prep_output + ']], '
    return dag_output, 3, prep_output


def fastICA(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"FastICA\": [[' + prep_input + '], [\"FastICA\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"algorithm\":\"' + alg[i + 2] + '\", '
    dag_output += '\"fun\":\"' + alg[i + 3] + '\", '
    dag_output += '\"max_iter\":' + alg[i + 4] + ', '
    dag_output += '\"tol\":' + alg[i + 5] + ', '
    dag_output += '\"whiten\":\"' + \
        alg[i + 6] + '\"}], [' + prep_output + ']], '
    return dag_output, 7, prep_output


def gaussian_random(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"GaussianRandomProjection\": [[' + \
        prep_input + '], [\"GaussianRandomProjection\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"eps\":' + alg[i + 2] + '}], [' + prep_output + ']], '
    return dag_output, 3, prep_output


def sparse_random(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"SparseRandomProjection\": [[' + \
        prep_input + '], [\"SparseRandomProjection\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"eps\":' + alg[i + 2] + ', '
    dag_output += '\"density\":' + alg[i + 3] + ', '
    dag_output += '\"dense_output\":\"' + \
        alg[i + 4] + '\"}], [' + prep_output + ']], '
    return dag_output, 5, prep_output


def feature_agglomeration(alg, i, prep_input):
    print(alg[i], '\n')
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"FeatureAgglomeration\": [[' + \
        prep_input + '], [\"FeatureAgglomeration\", '
    dag_output += '{\"n_clusters\":' + alg[i + 1] + ', '
    dag_output += '\"affinity\":\"' + alg[i + 2] + '\", '
    dag_output += '\"linkage\":\"' + alg[i + 3] + '\", '
    dag_output += '\"compute_full_tree\":\"' + \
        alg[i + 4] + '\"}], [' + prep_output + ']], '
    return dag_output, 5, prep_output


def RBF(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"RBFSampler\": [[' + prep_input + '], [\"RBFSampler\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"gamma\":' + alg[i + 2] + '}], [' + prep_output + ']], '
    return dag_output, 3, prep_output


def nystroem(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"Nystroem\": [[' + prep_input + '], [\"Nystroem\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"kernel\":\"' + alg[i + 2] + '\", '
    dag_output += '\"gamma\":' + alg[i + 3] + ', '
    dag_output += '\"degree\":' + alg[i + 4] + ', '
    dag_output += '\"coef0\":' + \
        alg[i + 5] + '}], [' + prep_output + ']], '
    return dag_output, 6, prep_output


def truncatedSVD(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"TruncatedSVD\": [[' + prep_input + '], [\"TruncatedSVD\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"n_iter\":' + alg[i + 2] + ', '
    dag_output += '\"tol\":' + alg[i + 3] + ', '
    dag_output += '\"algorithm\":\"' + \
        alg[i + 4] + '\"}], [' + prep_output + ']], '
    return dag_output, 5, prep_output


def binarizer(alg, i, prep_input):
    prep_output = '\"PRE:' + str(i) + '\"'
    dag_output = '\"Binarizer\": [[' + prep_input + '], [\"Binarizer\", '
    dag_output += '{\"threshold\":' + \
        alg[i + 1] + '}], [' + prep_output + ']], '
    return dag_output, 2, prep_output


# Classification methods

def mlp(alg, input_param, i):
    dag_output = '\"MLP\": [[' + input_param + '], [\"MLP\", '
    dag_output += '{\"learning_rate\":\"' + alg[i + 1] + '\", '
    dag_output += '\"learning_rate_init\":' + alg[i + 2] + ', '
    dag_output += '\"momentum\":' + alg[i + 3] + ', '
    dag_output += '\"max_iter\":' + alg[i + 4] + ', '
    dag_output += '\"activation\":\"' + alg[i + 5] + '\"}], []]}'

    return dag_output, 6


def log_reg(alg, input_param, i):
    dag_output = '\"LogisticRegression\": [[' + \
        input_param + '], [\"LogisticRegression\", '
    dag_output += '{\"penalty\":\"' + alg[i + 1] + '\", '
    dag_output += '\"tol\":' + alg[i + 2] + ', '
    dag_output += '\"C\":' + alg[i + 3] + ', '
    dag_output += '\"fit_intercept\":\"' + alg[i + 4] + '\", '
    dag_output += '\"max_iter\":' + alg[i + 5] + ', '
    dag_output += '\"warm_start\":\"' + alg[i + 6] + '\"}], []]}'

    return dag_output, 7


def dt(alg, input_param, i):
    dag_output = '\"DT\": [[' + input_param + '], [\"DT\", '
    dag_output += '{\"criterion\":\"' + alg[i + 1] + '\", '
    dag_output += '\"splitter\":\"' + alg[i + 2] + '\", '
    dag_output += '\"max_depth\":' + alg[i + 3] + ', '
    dag_output += '\"max_features\":\"' + alg[i + 4] + '\", '
    dag_output += '\"min_weight_fraction_leaf\":' + alg[i + 5] + ', '
    dag_output += '\"max_leaf_nodes\":' + alg[i + 6] + '}], []]}'

    return dag_output, 7


def svc(alg, input_param, i):
    dag_output = '\"SVC\": [[' + input_param + '], [\"SVC\", '
    dag_output += '{\"C\":' + alg[i + 1] + ', '
    dag_output += '\"kernel\":\"' + alg[i + 2] + '\", '
    dag_output += '\"degree\":' + alg[i + 3] + ', '
    dag_output += '\"gamma\":' + alg[i + 4] + ', '
    dag_output += '\"coef0\":' + alg[i + 5] + ', '
    dag_output += '\"probability\":\"' + alg[i + 6] + '\", '
    dag_output += '\"shrinking\":\"' + alg[i + 7] + '\", '
    dag_output += '\"decision_function_shape\":\"' + alg[i + 8] + '\", '
    dag_output += '\"tol\":' + alg[i + 9] + ', '
    dag_output += '\"max_iter\":' + alg[i + 10] + ', '
    dag_output += '\"class_weight\":\"' + alg[i + 11] + '\"}], []]}'

    return dag_output, 12


def NuSVC(alg, input_param, i):
    dag_output = '\"NuSVC\": [[' + input_param + '], [\"NuSVC\", '
    dag_output += '{\"nu\":' + alg[i + 1] + ', '
    dag_output += '\"kernel\":\"' + alg[i + 2] + '\", '
    dag_output += '\"degree\":' + alg[i + 3] + ', '
    dag_output += '\"gamma\":' + alg[i + 4] + ', '
    dag_output += '\"coef0\":' + alg[i + 5] + ', '
    dag_output += '\"probability\":\"' + alg[i + 6] + '\", '
    dag_output += '\"shrinking\":\"' + alg[i + 7] + '\", '
    dag_output += '\"decision_function_shape\":\"' + alg[i + 8] + '\", '
    dag_output += '\"tol\":' + alg[i + 9] + ', '
    dag_output += '\"max_iter\":' + alg[i + 10] + ', '
    dag_output += '\"class_weight\":\"' + alg[i + 11] + '\"}], []]}'

    return dag_output, 12


def gnb(alg, input_param, i):
    dag_output = '\"gaussianNB\": [[' + input_param + '], [\"gaussianNB\", '
    dag_output += '{}], []]}'

    return dag_output, 1


def lda(alg, input_param, i):
    dag_output = '\"LDA\": [[' + input_param + '], [\"LDA\", '
    dag_output += '{\"n_components\":' + alg[i + 1] + ', '
    dag_output += '\"tol\":' + alg[i + 2] + '}], []]}'

    return dag_output, 3


def qda(alg, input_param, i):
    dag_output = '\"QDA\": [[' + input_param + '], [\"QDA\", '
    dag_output += '{\"reg_param\":' + alg[i + 1] + ', '
    dag_output += '\"tol\":' + alg[i + 2] + '}], []]}'

    return dag_output, 3


def sgd(alg, input_param, i):
    dag_output = '\"SGD\": [[' + input_param + '], [\"SGD\", '
    dag_output += '{\"penalty\":\"' + alg[i + 1] + '\", '
    dag_output += '\"tol\":' + alg[i + 2] + ', '
    dag_output += '\"max_iter\":' + alg[i + 3] + ', '
    dag_output += '\"loss\":\"' + alg[i + 4] + '\", '
    dag_output += '\"warm_start\":\"' + alg[i + 5] + '\"}], []]}'

    return dag_output, 6


def perceptron(alg, input_param, i):
    dag_output = '\"Perceptron\": [[' + input_param + '], [\"Perceptron\", '
    dag_output += '{\"penalty\":\"' + alg[i + 1] + '\", '
    dag_output += '\"tol\":' + alg[i + 2] + ', '
    dag_output += '\"max_iter\":' + alg[i + 3] + ', '
    dag_output += '\"warm_start\":\"' + alg[i + 4] + '\"}], []]}'

    return dag_output, 5


def bernoulli(alg, input_param, i):
    dag_output = '\"BernoulliNB\": [[' + input_param + '], [\"BernoulliNB\", '
    dag_output += '{\"binarize\":' + alg[i + 1] + ', '
    dag_output += '\"alpha\":' + alg[i + 2] + ', '
    dag_output += '\"fit_prior\":\"' + alg[i + 3] + '\"}], []]}'

    return dag_output, 4


def multinomialNB(alg, input_param, i):
    dag_output = '\"MultinomialNB\": [[' + \
        input_param + '], [\"MultinomialNB\", '
    dag_output += '{\"alpha\":' + alg[i + 1] + ', '
    dag_output += '\"fit_prior\":\"' + alg[i + 2] + '\"}], []]}'

    return dag_output, 3


def complementNB(alg, input_param, i):
    dag_output = '\"ComplementNB\": [[' + \
        input_param + '], [\"ComplementNB\", '
    dag_output += '{\"alpha\":' + alg[i + 1] + ', '
    dag_output += '\"fit_prior\":\"' + alg[i + 2] + '\", '
    dag_output += '\"norm\":\"' + alg[i + 3] + '\"}], []]}'

    return dag_output, 4


def knn(alg, input_param, i):
    dag_output = '\"KNearestNeighbors\": [[' + \
        input_param + '], [\"KNearestNeighbors\", '
    dag_output += '{\"n_neighbors\":' + alg[i + 1] + ', '
    dag_output += '\"weights\":\"' + alg[i + 2] + '\", '
    dag_output += '\"algorithm\":\"' + alg[i + 3] + '\", '
    dag_output += '\"leaf_size\":' + alg[i + 4] + ', '
    dag_output += '\"p\":' + alg[i + 5] + ', '
    dag_output += '\"metric\":\"' + alg[i + 6] + '\"}], []]}'

    return dag_output, 7


def radius(alg, input_param, i):
    dag_output = '\"RadiusNeighbors\": [[' + \
        input_param + '], [\"RadiusNeighbors\", '
    dag_output += '{\"radius\":' + alg[i + 1] + ', '
    dag_output += '\"weights\":\"' + alg[i + 2] + '\", '
    dag_output += '\"algorithm\":\"' + alg[i + 3] + '\", '
    dag_output += '\"leaf_size\":' + alg[i + 4] + ', '
    dag_output += '\"p\":' + alg[i + 5] + ', '
    dag_output += '\"metric\":\"' + alg[i + 6] + '\"}], []]}'

    return dag_output, 7


def centroid(alg, input_param, i):
    dag_output = '\"Centroid\": [[ ' + input_param + '], [\"Centroid\", '
    dag_output += '{\"shrink_threshold\":' + alg[i + 1] + ', '
    dag_output += '\"metric\":\"' + alg[i + 2] + '\"}], []]}'

    return dag_output, 3


def ridge(alg, input_param, i):
    dag_output = '\"Ridge\": [[' + input_param + '], [\"Ridge\", '
    dag_output += '{\"alpha\":' + alg[i + 1] + ', '
    dag_output += '\"max_iter\":' + alg[i + 2] + ', '
    dag_output += '\"copy_X\":\"' + alg[i + 3] + '\", '
    dag_output += '\"solver\":\"' + alg[i + 4] + '\", '
    dag_output += '\"tol\":' + alg[i + 5] + ', '
    dag_output += '\"normalize\":\"' + alg[i + 6] + '\", '
    dag_output += '\"fit_intercept\":\"' + alg[i + 7] + '\"}], []]}'

    return dag_output, 8


def ridgeCCV(alg, input_param, i):
    dag_output = '\"RidgeCCV\": [[' + input_param + '], [\"RidgeCCV\", '
    dag_output += '{\"cv\":' + alg[i + 1] + ', '
    dag_output += '\"normalize\":\"' + alg[i + 2] + '\", '
    dag_output += '\"fit_intercept\":\"' + alg[i + 3] + '\"}], []]}'

    return dag_output, 4


def extraTree(alg, input_param, i):
    dag_output = '\"ExtraTree\": [[' + input_param + '], [\"ExtraTree\", '
    dag_output += '{\"criterion\":\"' + alg[i + 1] + '\", '
    dag_output += '\"splitter\":\"' + alg[i + 2] + '\", '
    dag_output += '\"class_weight\":\"' + alg[i + 3] + '\", '
    dag_output += '\"max_features\":\"' + alg[i + 4] + '\", '
    dag_output += '\"max_depth\":' + alg[i + 5] + ', '
    dag_output += '\"min_weight_fraction_leaf\":' + alg[i + 6] + ', '
    dag_output += '\"max_leaf_nodes\":' + alg[i + 7] + '}], []]}'

    return dag_output, 8


def randomForest(alg, input_param, i):
    dag_output = '\"RandomForest\": [[' + \
        input_param + '], [\"RandomForest\", '
    dag_output += '{\"criterion\":\"' + alg[i + 1] + '\", '
    dag_output += '\"bootstrap\":\"' + alg[i + 2] + '\", '
    dag_output += '\"oob_score\":\"' + alg[i + 3] + '\", '
    dag_output += '\"class_weight\":\"' + alg[i + 4] + '\", '
    dag_output += '\"n_estimators\":' + alg[i + 5] + ', '
    dag_output += '\"warm_start\":\"' + alg[i + 6] + '\", '
    dag_output += '\"max_features\":\"' + alg[i + 7] + '\", '
    dag_output += '\"max_depth\":' + alg[i + 8] + ', '
    dag_output += '\"min_weight_fraction_leaf\":' + alg[i + 9] + ', '
    dag_output += '\"max_leaf_nodes\":' + alg[i + 10] + '}], []]}'

    return dag_output, 11


def extraTrees(alg, input_param, i):
    dag_output = '\"ExtraTrees\": [[' + input_param + '], [\"ExtraTrees\", '
    dag_output += '{\"criterion\":\"' + alg[i + 1] + '\", '
    dag_output += '\"bootstrap\":\"' + alg[i + 2] + '\", '
    #dag_output += '\"oob_score\":\"' + alg[i + 3] + '\", '
    dag_output += '\"class_weight\":\"' + alg[i + 4] + '\", '
    dag_output += '\"n_estimators\":' + alg[i + 5] + ', '
    dag_output += '\"warm_start\":\"' + alg[i + 6] + '\", '
    dag_output += '\"max_features\":\"' + alg[i + 7] + '\", '
    dag_output += '\"max_depth\":' + alg[i + 8] + ', '
    dag_output += '\"min_weight_fraction_leaf\":' + alg[i + 9] + ', '
    dag_output += '\"max_leaf_nodes\":' + alg[i + 10] + '}], []]}'

    return dag_output, 11


def ada(alg, input_param, i):
    dag_output = '\"AdaBoost\": [[' + input_param + '], [\"AdaBoost\", '
    dag_output += '{\"algorithm\":\"' + alg[i + 1] + '\", '
    dag_output += '\"n_estimators\":' + alg[i + 2] + ', '
    dag_output += '\"learning_rate\":' + alg[i + 3] + '}], []]}'

    return dag_output, 4


def gb(alg, input_param, i):
    dag_output = '\"GradientBoosting\": [[' + \
        input_param + '], [\"GradientBoosting\", '
    dag_output += '{\"loss\":\"' + alg[i + 1] + '\", '
    dag_output += '\"tol\":\"' + alg[i + 2] + '\", '
    dag_output += '\"learning_rate\":' + alg[i + 3] + ', '
    dag_output += '\"presort\":\"' + alg[i + 4] + '\", '
    dag_output += '\"n_estimators\":' + alg[i + 5] + ', '
    dag_output += '\"warm_start\":\"' + alg[i + 6] + '\", '
    dag_output += '\"max_features\":\"' + alg[i + 7] + '\", '
    dag_output += '\"max_depth\":' + alg[i + 8] + ', '
    dag_output += '\"min_weight_fraction_leaf\":' + alg[i + 9] + ', '
    dag_output += '\"max_leaf_nodes\":' + alg[i + 10] + '}], []]}'

    return dag_output, 11


# Dictionary to store preprocessing methods names
preproc_methods = {
    'SimpleImputer': simp,
    'Normalizer': norm,
    'MinMaxScaler': minmax,
    'MaxAbsScaler': maxabs,
    'RobustScaler': robust,
    'StandardScaler': standard,
    'VarianceThreshold': variance,
    'SelectKBest': kBest,
    'PCA': PCA,
    'IncrementalPCA': incrementalPCA,
    'FastICA': fastICA,
    'GaussianRandomProjection': gaussian_random,
    'SparseRandomProjection': sparse_random,
    'FeatureAgglomeration': feature_agglomeration,
    'RBFSampler': RBF,
    'Nystroem': nystroem,
    'TruncatedSVD': truncatedSVD,
    'Binarizer': binarizer
}

# Dictionary to store classification methods names
classif_methods = {
    'gaussianNB': gnb,
    'BernoulliNB': bernoulli,
    'MultinomialNB': multinomialNB,
    'ComplementNB': complementNB,
    'SVC': svc,
    'NuSVC': NuSVC,
    'LogisticRegression': log_reg,
    'Perceptron': perceptron,
    'MLP': mlp,
    'SGD': sgd,
    'LDA': lda,
    'QDA': qda,
    'KNearestNeighbors': knn,
    'RadiusNeighbors': radius,
    'Centroid': centroid,
    'Ridge': ridge,
    'RidgeCCV': ridgeCCV,
    'DT': dt,
    'ExtraTree': extraTree,
    'RandomForest': randomForest,
    'ExtraTrees': extraTrees,
    'AdaBoost': ada,
    'GradientBoosting': gb
}
