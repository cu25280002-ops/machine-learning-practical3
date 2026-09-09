import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
model2 = LinearRegression().fit(X_poly2, y)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
model3 = LinearRegression().fit(X_poly3, y)
X_grid = np.arange(X['Years_Experience'].min(), X['Years_Experience'].max(), 0.1).reshape(-1, 1)
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(X_grid, model2.predict(poly2.transform(X_grid)), color='blue', label='Degree 2')
plt.plot(X_grid, model3.predict(poly3.transform(X_grid)), color='green', label='Degree 3')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Polynomial Regression Curves')
plt.legend()
plt.show()