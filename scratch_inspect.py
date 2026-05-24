import pandas as pd

df = pd.read_excel('C:/Users/rohit/.gemini/antigravity/scratch/group_project/data/pv_data_sample.xlsx')
print("Columns:", df.columns.tolist())
if 'Condition:(1PS)/(2MM)' in df.columns:
    print("Unique conditions:", df['Condition:(1PS)/(2MM)'].unique())
    print("Value counts:\n", df['Condition:(1PS)/(2MM)'].value_counts())
if 'Label' in df.columns:
    print("Unique Labels:", df['Label'].unique())
