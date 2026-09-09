import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_tr = poly.fit_transform(X_train)
    X_te = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_tr, y_train)
    score = r2_score(y_test, model.predict(X_te))
    print(f"Degree {degree} R2 Score: {score:.4f}")