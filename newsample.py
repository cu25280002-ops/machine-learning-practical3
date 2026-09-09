import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('salary_data.csv')
X = df[['Years_Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
new_experience = pd.DataFrame({'Years_Experience': [5.5, 8.0, 12.0]})
predicted_salary = model.predict(new_experience)
for exp, sal in zip(new_experience['Years_Experience'], predicted_salary):
    print(f"Experience: {exp} years -> Predicted Salary: {sal:.2f}")