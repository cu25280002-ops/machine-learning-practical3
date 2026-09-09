import pandas as pd
import joblib
loaded_model = joblib.load('regression_model.pkl')
new_data = pd.DataFrame({'Years_Experience': [4.5, 7.0]})
predictions = loaded_model.predict(new_data)
for exp, pred in zip([4.5, 7.0], predictions):
    print(f"Experience: {exp} years -> Predicted Salary: {pred:.2f}")