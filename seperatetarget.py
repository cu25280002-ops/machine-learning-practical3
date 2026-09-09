import pandas as pd
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
print("Input Variable (X):")
print(X.head())
print("\nOutput Variable (y):")
print(y.head())