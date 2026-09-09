import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
model = LinearRegression()
print("Linear Regression model initialized successfully:")
print(model)