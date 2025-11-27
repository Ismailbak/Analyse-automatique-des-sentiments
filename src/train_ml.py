"""Train classical ML models: Logistic Regression, SVM, Random Forest."""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def train_logreg(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def train_svm(X, y):
    clf = LinearSVC(max_iter=10000)
    clf.fit(X, y)
    return clf


def train_rf(X, y):
    clf = RandomForestClassifier(n_estimators=200)
    clf.fit(X, y)
    return clf


def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    return classification_report(y_test, preds)


def save_model(clf, path: str):
    joblib.dump(clf, path)
