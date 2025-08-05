# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def add_derived_features(df):
    """
    Derive new features from existing output features and drop near-constant ones.
    """
    print("\n🧪 Adding derived features...")

    df['p_ratio'] = df['Pmax'] / (df['Vmax'] * df['Imax'])
    df['voc_vmax_ratio'] = df['Voc'] / df['Vmax']
    df['isc_imax_ratio'] = df['Isc'] / df['Imax']
    df['voc_isc_ratio'] = df['Voc'] / df['Isc']
    df['pmax_voc_ratio'] = df['Pmax'] / df['Voc']
    df['pmax_isc_ratio'] = df['Pmax'] / df['Isc']
    df['vmax_imax_ratio'] = df['Vmax'] / df['Imax']
    df['voc_minus_vmax'] = df['Voc'] - df['Vmax']
    df['isc_minus_imax'] = df['Isc'] - df['Imax']
    df['pmax_minus_voc_isc'] = df['Pmax'] - (df['Voc'] * df['Isc'])
    df['pmax_norm_mean'] = df['Pmax'] / df['Pmax'].mean()
    df['vmax_voc_ratio'] = df['Vmax'] / df['Voc']
    df['imax_isc_ratio'] = df['Imax'] / df['Isc']
    df['voc_isc_pmax'] = (df['Voc'] * df['Isc']) / df['Pmax']
    df['sqrt_pmax'] = df['Pmax'] ** 0.5

    derived_cols = [
        'p_ratio', 'voc_vmax_ratio', 'isc_imax_ratio', 'voc_isc_ratio',
        'pmax_voc_ratio', 'pmax_isc_ratio', 'vmax_imax_ratio',
        'voc_minus_vmax', 'isc_minus_imax', 'pmax_minus_voc_isc',
        'pmax_norm_mean', 'vmax_voc_ratio', 'imax_isc_ratio',
        'voc_isc_pmax', 'sqrt_pmax'
    ]

    keep_cols = []
    for col in derived_cols:
        std = df[col].std()
        if std > 0.01:  # drop near-constant
            keep_cols.append(col)
        else:
            print(f"⚠️ Dropping near-constant feature: {col} (std={std:.5f})")

    print(f"\n✅ Derived features kept: {keep_cols}")
    return df

def preprocess_pv_data(df):
    """
    Add derived features, scale numeric features.
    """
    df = add_derived_features(df)

    # Select input features
    input_features = ['Temp', 'Irradiance', 'Resistance', 'Row', 'Col'] \
                     + [col for col in df.columns if col in [
                         'p_ratio', 'voc_vmax_ratio', 'isc_imax_ratio',
                         'voc_isc_ratio', 'pmax_voc_ratio', 'pmax_isc_ratio',
                         'vmax_imax_ratio', 'voc_minus_vmax', 'isc_minus_imax',
                         'pmax_minus_voc_isc', 'pmax_norm_mean', 'vmax_voc_ratio',
                         'imax_isc_ratio', 'voc_isc_pmax', 'sqrt_pmax'
                     ]]
    
    # Output features
    output_features = ['Pmax', 'Vmax', 'Imax', 'Voc', 'Isc']

    X = df[input_features]
    y = df[output_features]

    print("\n🔍 Input features used:")
    print(X.columns)

    # Scale inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train and test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Example: test using sample CSV or Excel
    print("📦 feature_engineering.py module loaded successfully.")
