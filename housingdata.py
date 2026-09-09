import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv('housing_data.csv')
print("Housing Dataset Loaded:")
print(df.head())
X = df[['Area_SqFt', 'Bedrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
print("Housing Price Model R2 Score:", r2_score(y_test, predictions))