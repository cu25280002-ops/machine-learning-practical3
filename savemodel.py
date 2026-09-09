import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
model = LinearRegression().fit(X, y)
joblib.dump(model, 'regression_model.pkl')
print("Model saved to 'regression_model.pkl'.")