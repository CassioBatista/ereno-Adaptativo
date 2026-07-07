"""GenericClassifiers equivalent — sklearn classifier wrappers matching Weka classifiers."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


class ClassifierExtended:
    """Wraps a sklearn classifier with a display name."""

    def __init__(self, classifier, name: str, enabled: bool = True):
        self.classifier = classifier
        self.name = name
        self.enabled = enabled

    def get_classifier(self):
        return self.classifier

    def get_classifier_name(self) -> str:
        return self.name

    def reset(self):
        return self.classifier.__class__(**self.classifier.get_params())


# Weka → sklearn equivalences
RANDOM_TREE = ClassifierExtended(
    DecisionTreeClassifier(splitter="random", max_features="sqrt", random_state=42),
    "RandomTree",
)
J48 = ClassifierExtended(
    DecisionTreeClassifier(criterion="entropy", random_state=42),
    "J48",
)
REP_TREE = ClassifierExtended(
    DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, random_state=42),
    "REPTree",
)
NAIVE_BAYES   = ClassifierExtended(GaussianNB(), "NaiveBayes")
RANDOM_FOREST = ClassifierExtended(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "RandomForest",
)
KNN = ClassifierExtended(KNeighborsClassifier(n_neighbors=1), "KNN")
XGB = ClassifierExtended(
    XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "XGBoost",
)

all_classifiers = [RANDOM_TREE, J48, REP_TREE, NAIVE_BAYES, RANDOM_FOREST, XGB]
all_custom      = all_classifiers
