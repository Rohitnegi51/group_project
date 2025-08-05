import pandas as pd

def load_pv_data(file_path):
    """
    Load real PV dataset from Excel.
    Clean numeric columns: remove spaces and '+' signs, convert to float.
    """
    try:
        df = pd.read_excel(file_path)
        print("✅ Data loaded successfully!")
        print("Shape:", df.shape)

        # Columns that should be numeric
        numeric_cols = ['Pmax', 'Imax', 'Vmax', 'Voc', 'Isc', 'Temp', 'Irradiance', 'Resistance']

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(' ', '').str.replace('+', '', regex=False),
                errors='coerce'
            )

        # Clean column name for fault label
        fault_col = 'Condition:(1PS)/(2MM)'
        if fault_col not in df.columns:
            raise ValueError(f"Column '{fault_col}' not found in dataset.")

        # Map the fault column to labels
        label_map = {1: 'PS', 2: 'MM'}
        df['Label'] = df[fault_col].map(label_map)

        if df['Label'].isnull().any():
            raise ValueError("Some rows have unknown or missing labels in fault column.")

        print("\n✅ Numeric columns cleaned.")
        print("✅ Fault labels mapped. Sample:")
        print(df[['Condition_ID', 'Condition_Name', fault_col, 'Label']].head())
        
        print("\n✅ Numeric columns cleaned. Sample data:")
        print(df.head())

        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

if __name__ == "__main__":
    # For testing: update path to your Excel
    file_path = "data/pv_data_sample.xlsx"
    df = load_pv_data(file_path)

    if df is not None:
        print("\nUnique test conditions:")
        print(df['Condition_Name'].unique())

        print("\nAverage Pmax by Condition:")
        print(df.groupby('Condition_Name')['Pmax'].mean())

        print("\nData types after cleaning:")
        print(df.dtypes)
