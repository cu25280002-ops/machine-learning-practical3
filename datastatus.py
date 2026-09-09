import pandas as pd
df = pd.read_csv('salary_data.csv')
print("Info")
df.info()
print("\nStatistics")
print(df.describe())