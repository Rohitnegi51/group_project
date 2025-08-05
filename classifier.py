# src/classifier.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_pv_data
from feature_engineering import create_features
from feature_selection import reptile_search_algorithm

def train_and_evaluate_classifier(X, y, selected_idx, test_size=0.2, random_state=42):
    """
    Train and evaluate classifier on selected features.
    """
    X_selected = X[:, selected_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = clf.score(X_test, y_test)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f}")

    return clf

if __name__ == "__main__":
    file_path = "../data/pv_data_sample.xlsx"
    df = load_pv_data(file_path)
    df_fe = create_features(df)

    # Define input and output columns
    output_cols = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc']
    input_cols = [col for col in df_fe.columns if col not in output_cols + ['Condition_ID', 'Condition_Name', 'Row', 'Col']]

    X = df_fe[input_cols].to_numpy()
    y = df_fe['Condition_ID'].to_numpy()

    # Run feature selection
    selected_idx = reptile_search_algorithm(X, y)

    # Train and evaluate
    clf = train_and_evaluate_classifier(X, y, selected_idx)
