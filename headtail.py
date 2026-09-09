import pandas as pd
df = pd.read_csv('salary_data.csv')
print("First 5 Records")
print(df.head(5))
print("\nLast 5 Records")
print(df.tail(5))