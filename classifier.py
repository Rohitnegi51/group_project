# src/classifier.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_pv_data
from feature_engineering import add_derived_features
from feature_selection import reptile_search_algorithm

def train_and_evaluate_classifier(X, y, selected_idx, test_size=0.2, random_state=1):
    """
    Train and evaluate classifier on selected features.
    """
    X_selected = X[:, selected_idx]

    # Custom Stratified Split to handle tiny classes (e.g. Normal = 2)
    X_train, X_test, y_train, y_test = [], [], [], []
    np.random.seed(random_state)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        n_test = max(1, int(len(idx) * test_size))
        if len(idx) == 1:
            n_test = 0  # if only 1 sample exists, put it in train
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        
        X_train.extend(X_selected[train_idx])
        y_train.extend(y[train_idx])
        X_test.extend(X_selected[test_idx])
        y_test.extend(y[test_idx])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = clf.score(X_test, y_test)
    print(f"\n Test Accuracy: {accuracy:.4f}")

    return clf

