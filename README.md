# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and create the employee dataset.
2. Separate the input feature (Experience) and output variable (Salary).
3. Split the dataset into training and testing data and train the Decision Tree Regressor model.
4. Predict the employee salary and evaluate the model using Mean Squared Error. 

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create employee dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [20000, 25000, 30000, 35000, 40000,
               45000, 50000, 55000, 60000, 65000]
}

df = pd.DataFrame(data)

# Separate input and output
X = df[['Experience']]
y = df['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict salary
y_pred = model.predict(X_test)

# Calculate error and accuracy score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Actual Salary:", y_test.values)
print("Predicted Salary:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Predict salary for a new employee
new_employee = pd.DataFrame([[6]], columns=['Experience'])

predicted_salary = model.predict(new_employee)

print("Predicted Salary for 6 Years Experience:",
      predicted_salary[0])

# Plot graph
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.scatter(X, model.predict(X), color='red',
            label='Predicted Salary')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Decision Tree Regressor")
plt.legend()
plt.show()
```

## Output:
<img width="982" height="692" alt="image" src="https://github.com/user-attachments/assets/f5ecb5d8-f2aa-4c8d-966a-b965c0b704aa" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
