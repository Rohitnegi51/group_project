import pandas as pd

def load_pv_data(file_path):
    """
    Load real PV dataset from Excel.
    Clean numeric columns: remove spaces and '+' signs, convert to float.
    """
    try:
        df = pd.read_excel(file_path)
        print(" Data loaded successfully!")
        print("Shape:", df.shape)

        # Columns that should be numeric
        numeric_cols = ['Pmax', 'Imax', 'Vmax', 'Voc', 'Isc', 'Temp', 'Irradiance', 'Resistance']

        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(' ', '').str.replace('+', '', regex=False),
                errors='coerce'
            )

        # Find the column containing 'Condition:' dynamically
        fault_col = [col for col in df.columns if 'Condition:' in col]
        if not fault_col:
            raise ValueError("No column containing 'Condition:' found in dataset.")
        fault_col = fault_col[0]

        # Map the fault column to labels
        label_map = {1: 'PS', 2: 'MM', 3: 'Normal', 4: 'CrossString'}
        df['Label'] = df[fault_col].map(label_map)

        if df['Label'].isnull().any():
            raise ValueError("Some rows have unknown or missing labels in fault column.")

        print("\n Numeric columns cleaned.")
        print(" Fault labels mapped. Sample:")
        print(df[['Condition_ID', 'Condition_Name', fault_col, 'Label']].head())
        


        return df
    except Exception as e:
        print(f" Failed to load data: {e}")
        return None

