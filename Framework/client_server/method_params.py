import custom_models

from sklearn import impute
from sklearn import cluster
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import random_projection
from sklearn import kernel_approximation

from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import neural_network
from sklearn import discriminant_analysis

model_names = {
    "SimpleImputer":
        custom_models.make_transformer(impute.SimpleImputer),
    "Normalizer":
        custom_models.make_transformer(preprocessing.Normalizer),
    "MinMaxScaler":
        custom_models.make_transformer(preprocessing.MinMaxScaler),
    "MaxAbsScaler":
        custom_models.make_transformer(preprocessing.MaxAbsScaler),
    "RobustScaler":
        custom_models.make_transformer(preprocessing.RobustScaler),
    "StandardScaler":
        custom_models.make_transformer(preprocessing.StandardScaler),
    "VarianceThreshold":
        custom_models.make_transformer(feature_selection.VarianceThreshold),
    "SelectKBest":
        custom_models.make_transformer(feature_selection.SelectKBest),
    "SelectPercentile":
        custom_models.make_transformer(feature_selection.SelectPercentile),
    "SelectFpr":
        custom_models.make_transformer(feature_selection.SelectFpr),
    "SelectFwe":
        custom_models.make_transformer(feature_selection.SelectFwe),
    "SelectFdr":
        custom_models.make_transformer(feature_selection.SelectFdr),
    "RFE":
        custom_models.make_transformer(feature_selection.RFE),
    "RFECV":
        custom_models.make_transformer(feature_selection.RFECV),
    "SelectFromModel":
        custom_models.make_transformer(feature_selection.SelectFromModel),
    "PCA":
        custom_models.make_transformer(decomposition.PCA),
    "IncrementalPCA":
        custom_models.make_transformer(decomposition.IncrementalPCA),
    "FastICA":
        custom_models.make_transformer(decomposition.FastICA),
    "GaussianRandomProjection":
        custom_models.make_transformer(random_projection.
                                       GaussianRandomProjection),
    "SparseRandomProjection":
        custom_models.make_transformer(random_projection.
                                       SparseRandomProjection),
    "FeatureAgglomeration":
        custom_models.make_transformer(cluster.FeatureAgglomeration),
    "RBFSampler":
        custom_models.make_transformer(kernel_approximation.RBFSampler),
    "Nystroem":
        custom_models.make_transformer(kernel_approximation.Nystroem),
    "TruncatedSVD":
        custom_models.make_transformer(decomposition.TruncatedSVD),
    "PolynomialFeatures":
        custom_models.make_transformer(preprocessing.PolynomialFeatures),
    "Binarizer":
        custom_models.make_transformer(preprocessing.Binarizer),
    "OneHotEncoder":
        custom_models.make_transformer(preprocessing.OneHotEncoder),
    "gaussianNB":
        custom_models.make_predictor(naive_bayes.GaussianNB),
    "BernoulliNB":
        custom_models.make_predictor(naive_bayes.BernoulliNB),
    "MultinomialNB":
        custom_models.make_predictor(naive_bayes.MultinomialNB),
    "ComplementNB":
        custom_models.make_predictor(naive_bayes.ComplementNB),
    "SVC":
        custom_models.make_predictor(svm.SVC),
    "NuSVC":
        custom_models.make_predictor(svm.NuSVC),
    "LogisticRegression":
        custom_models.make_predictor(linear_model.LogisticRegression),
    "Perceptron":
        custom_models.make_predictor(linear_model.Perceptron),
    "MLP":
        custom_models.make_predictor(neural_network.MLPClassifier),
    "SGD":
        custom_models.make_predictor(linear_model.SGDClassifier),
    "LDA":
        custom_models.make_predictor(discriminant_analysis.
                                     LinearDiscriminantAnalysis),
    "QDA":
        custom_models.make_predictor(discriminant_analysis.
                                     QuadraticDiscriminantAnalysis),
    "KNearestNeighbors":
        custom_models.make_predictor(neighbors.KNeighborsClassifier),
    "RadiusNeighbors":
        custom_models.make_predictor(neighbors.RadiusNeighborsClassifier),
    "Centroid":
        custom_models.make_predictor(neighbors.NearestCentroid),
    "Ridge":
        custom_models.make_predictor(linear_model.RidgeClassifier),
    "RidgeCCV":
        custom_models.make_predictor(linear_model.RidgeClassifierCV),
    "DT":
        custom_models.make_predictor(tree.DecisionTreeClassifier),
    "ExtraTree":
        custom_models.make_predictor(tree.ExtraTreeClassifier),
    "RandomForest":
        custom_models.make_predictor(ensemble.RandomForestClassifier),
    "ExtraTrees":
        custom_models.make_predictor(ensemble.ExtraTreesClassifier),
    "AdaBoost":
        custom_models.make_predictor(ensemble.AdaBoostClassifier),
    "GradientBoosting":
        custom_models.make_predictor(ensemble.GradientBoostingClassifier),
}
