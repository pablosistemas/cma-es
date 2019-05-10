import custom_models
from sklearn import svm
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import decomposition
from sklearn import neural_network
from sklearn import feature_selection
from sklearn import discriminant_analysis

model_names = {
    "PCA":
        custom_models.make_transformer(decomposition.PCA),
    "kBest":
        custom_models.make_transformer(feature_selection.SelectKBest),
    "kMeans":
        custom_models.make_transformer(custom_models.KMeansSplitter),
    "copy":
        [],
    "SVC":
        custom_models.make_predictor(svm.SVC),
    "logR":
        custom_models.make_predictor(linear_model.LogisticRegression),
    "Perceptron":
        custom_models.make_predictor(linear_model.Perceptron),
    "SGD":
        custom_models.make_predictor(linear_model.SGDClassifier),
    "PAC":
        custom_models.make_predictor(linear_model.PassiveAggressiveClassifier),
    "LDA":
        custom_models.make_predictor(discriminant_analysis.
                                     LinearDiscriminantAnalysis),
    "QDA":
        custom_models.make_predictor(discriminant_analysis.
                                     QuadraticDiscriminantAnalysis),
    "MLP":
        custom_models.make_predictor(neural_network.MLPClassifier),
    "gaussianNB":
        custom_models.make_predictor(naive_bayes.GaussianNB),
    "DT":
        custom_models.make_predictor(tree.DecisionTreeClassifier),
    "union":
        custom_models.Voter,
    "vote":
        custom_models.Voter,
    "stacker":
        custom_models.Stacker,
    "booster":
        custom_models.Booster
}
