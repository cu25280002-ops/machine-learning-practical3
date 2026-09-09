import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience', 'Education_Level']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
raw_model = LinearRegression().fit(X_train, y_train)
raw_pred = raw_model.predict(X_test)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_te_scaled = scaler.transform(X_test)
scaled_model = LinearRegression().fit(X_tr_scaled, y_train)
scaled_pred = scaled_model.predict(X_te_scaled)
print("R2 Score without scaling:", r2_score(y_test, raw_pred))
print("R2 Score with scaling:", r2_score(y_test, scaled_pred))