# main.py

import pandas as pd
from data_loader import load_pv_data
from feature_engineering import add_derived_features
from feature_selection import reptile_search_algorithm, pso_feature_selection, som_ga_feature_selection
from classifier import train_and_evaluate_classifier
from sklearn.preprocessing import StandardScaler

def main():
    print("\n--- Step 1: Loading Data ---")
    df = load_pv_data("data/pv_data_sample.xlsx")
    if df is None:
        return

    print("\n--- Step 2: Feature Engineering ---")
    df_fe = add_derived_features(df)

    output_cols = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc', 'Label', 'Condition:(1PS)/(2MM)']
    input_cols = [col for col in df_fe.columns if col not in output_cols + ['Condition_ID', 'Condition_Name', 'Row', 'Col']]

    X = df_fe[input_cols].to_numpy()
    y = df_fe['Label'].to_numpy() # Target is 'Label' (PS, MM, Normal)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Step 3: Feature Selection & Evaluation Comparison ---")
    algorithms = {
        "CRSA (Chaotic Reptile Search)": reptile_search_algorithm,
        "PSO (Particle Swarm Optimization)": pso_feature_selection,
        "SOM-GA (Self-Organizing Maps + GA)": som_ga_feature_selection
    }

    for name, algo_func in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")
        
        # 1. Feature Selection
        selected_idx = algo_func(X_scaled, y)
        selected_feature_names = [input_cols[i] for i in selected_idx]
        
        print(f"Selected {len(selected_idx)} features:\n{selected_feature_names}")
        
        # 2. Train and Evaluate
        clf = train_and_evaluate_classifier(X_scaled, y, selected_idx)
        
    print(f"\n{'='*60}")
    print("Comparison Complete! Review the metrics above.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
