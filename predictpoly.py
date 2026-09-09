import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
new_data = pd.DataFrame({'Years_Experience': [3.0, 6.5, 9.0]})
new_data_poly = poly.transform(new_data)
predictions = model.predict(new_data_poly)
print(predictions)