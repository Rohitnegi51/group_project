import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

data = []
condition_id = 1

# We will generate 3 test cases (Conditions). Each case is a full 3x10 PV array grid (30 panels).
# Case 1: Normal Operation (Condition 3)
# Case 2: Partial Shading on Row 1 (Condition 1)
# Case 3: Mismatch on Col 5 (Condition 2)

conditions = [
    {'id': 1, 'name': 'Normal 1000W/m2', 'label': 3},
    {'id': 2, 'name': 'PS Row 1 600W/m2', 'label': 1},
    {'id': 3, 'name': 'MM Col 5 High Resistance', 'label': 2}
]

for cond in conditions:
    for r in range(1, 4):  # 3 Rows
        for c in range(1, 11):  # 10 Columns
            
            # Baseline normal physics
            irradiance = 1000
            temp = 25
            pmax = 250.0 + np.random.normal(0, 2)
            vmax = 30.0 + np.random.normal(0, 0.5)
            imax = 8.3 + np.random.normal(0, 0.1)
            voc = 37.0 + np.random.normal(0, 0.5)
            isc = 8.8 + np.random.normal(0, 0.1)
            resistance = 0.0
            
            # Inject Fault Physics based on Condition
            if cond['label'] == 1 and r == 1:
                # Partial Shading on Row 1: Low Irradiance, Current Drops
                irradiance = 600
                imax = 5.0 + np.random.normal(0, 0.1)
                isc = 5.3 + np.random.normal(0, 0.1)
                pmax = vmax * imax * 0.9 # Efficiency drop
                
            elif cond['label'] == 2 and c == 5:
                # String Mismatch on Col 5: High Resistance, Voltage Drops
                resistance = 2.5 + np.random.normal(0, 0.2)
                vmax = 24.0 + np.random.normal(0, 0.5)
                voc = 30.0 + np.random.normal(0, 0.5)
                pmax = vmax * imax * 0.85 # Efficiency drop
            
            data.append({
                'Condition_ID': cond['id'],
                'Condition:(1PS)/(2MM)/(3Normal)': cond['label'],
                'Condition_Name': cond['name'],
                'Row': r,
                'Col': c,
                'Pmax': round(pmax, 2),
                'Vmax': round(vmax, 2),
                'Imax': round(imax, 2),
                'Voc': round(voc, 2),
                'Isc': round(isc, 2),
                'Temp': temp,
                'Irradiance': irradiance,
                'Resistance': round(resistance, 2)
            })

df = pd.DataFrame(data)
output_path = 'data/mock_3x10_grid_test.xlsx'
df.to_excel(output_path, index=False)
print(f"Successfully generated {output_path} with {len(df)} rows!")
