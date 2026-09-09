import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience', 'Education_Level']]
y = df['Salary']
model = LinearRegression()
print("Multiple Linear Regression model initialized.")