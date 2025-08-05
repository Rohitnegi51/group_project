# main.py

from data_loader import load_data
from feature_engineering import extract_features, normalize_features
from feature_selection import rsa_feature_selection
from classifier import train_and_evaluate_models

def main():
    # Step 1: Load dataset
    df = load_data("data/pv_data_sample.xlsx")

    # Step 2: Feature engineering
    features = extract_features(df)
    features_norm = normalize_features(features)

    # Step 3: Feature selection using RSA with Chaotic Logistic Map
    selected_indices = rsa_feature_selection(features_norm, df["Label"])
    selected_features = features_norm.iloc[:, selected_indices]

    # Step 4: Train and evaluate models
    train_and_evaluate_models(selected_features, df["Label"])

if __name__ == "__main__":
    main()
