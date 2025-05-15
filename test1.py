import pandas as pd


df = pd.read_parquet('C:\\temp\\validation-00000-of-00001.parquet')
df.to_csv('C:\\temp\\validation.csv')
print(df)
