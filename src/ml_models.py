from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from ml_utils import read_and_extract_features, print_metrics_binary, Reader
import numpy as np
import argparse

DATA = "../data/preprocessed_data/"
MODEL = "LR"

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="choose machine learning model", type=str)
args = parser.parse_args()
MODEL = args.model




def main():
    print('Machine learning model is ' + MODEL)
    #dataset reader
    train_reader = Reader(dataset_dir=DATA + "train", listfile=DATA + "train_listfile.csv")
    val_reader = Reader(dataset_dir=DATA + "train", listfile=DATA + "val_listfile.csv")
    test_reader = Reader(dataset_dir=DATA + "test", listfile=DATA + "test_listfile.csv")                

    print('Reading data and extracting features ...')
    (train_X, train_y) = read_and_extract_features(train_reader)
    (val_X, val_y) = read_and_extract_features(val_reader)
    (test_X, test_y) = read_and_extract_features(test_reader)

    print('Imputing missing values ...')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    #build classifier
    if MODEL == "LR":
        clf = LogisticRegression(penalty="l2", C=0.001, solver='lbfgs')
    elif MODEL == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif MODEL == "SVM":
        clf = SVC(C=1.0, kernel="rbf", probability=True)
    elif MODEL == "DT":
        clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1)
    elif MODEL == "RF":
        clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2)
    elif MODEL == "ADA":
        clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
    clf.fit(train_X, train_y)

    #print result in terminal
    print('\nResults on train set')
    print_metrics_binary(train_y, clf.predict_proba(train_X))

    print('\nResults on eval set')
    print_metrics_binary(val_y, clf.predict_proba(val_X))

    prediction = clf.predict_proba(test_X)[:, 1]
    print('\nResults on test set')
    print_metrics_binary(test_y, prediction)

if __name__ == '__main__':
    main()