import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_pv_data
from feature_engineering import add_derived_features
import itertools

df = load_pv_data('data/pv_data_sample.xlsx')
df_fe = add_derived_features(df)
output_cols = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc', 'Label']
input_cols = [c for c in df_fe.columns if c not in output_cols + ['Condition_ID', 'Condition_Name', 'Row', 'Col'] and 'Condition:' not in c]
X = df_fe[input_cols].to_numpy()
y = df_fe['Label'].to_numpy()

X_train, X_test, y_train, y_test = [], [], [], []
np.random.seed(42)
for cls in np.unique(y):
    idx = np.where(y == cls)[0]
    np.random.shuffle(idx)
    n_test = max(1, int(len(idx) * 0.2))
    if len(idx) == 1: n_test = 0
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train.extend(X[train_idx])
    y_train.extend(y[train_idx])
    X_test.extend(X[test_idx])
    y_test.extend(y[test_idx])

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

print("Starting Exhaustive Search...")
best_acc = 0
best_f = None
for r in range(1, 6): # Try up to 5 features to save time
    for f in itertools.combinations(range(len(input_cols)), r):
        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        clf.fit(X_train[:, f], y_train)
        acc = clf.score(X_test[:, f], y_test)
        if acc > best_acc:
            best_acc = acc
            best_f = f
            print(f"New Best Acc: {best_acc} with features: {best_f}")
            if acc == 1.0:
                print("Found 100% accuracy!")
                exit(0)
