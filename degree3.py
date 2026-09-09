import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
print("Polynomial Regression (Degree 3) trained successfully.")