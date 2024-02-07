import pandas as pd

file_path = "data/raw_data.csv"
df = pd.read_csv(file_path)
print(df.head())
