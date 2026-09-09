import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression().fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
poly = PolynomialFeatures(degree=2)
poly_model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
poly_pred = poly_model.predict(poly.transform(X_test))
results = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Score'],
    'Linear': [
        mean_absolute_error(y_test, lin_pred),
        mean_squared_error(y_test, lin_pred),
        np.sqrt(mean_squared_error(y_test, lin_pred)),
        r2_score(y_test, lin_pred)
    ],
    'Polynomial (Deg 2)': [
        mean_absolute_error(y_test, poly_pred),
        mean_squared_error(y_test, poly_pred),
        np.sqrt(mean_squared_error(y_test, poly_pred)),
        r2_score(y_test, poly_pred)
    ]
})
print(results)