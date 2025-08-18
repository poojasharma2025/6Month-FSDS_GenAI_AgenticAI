import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
dataset = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\14th Aug-linear\Salary_Data.csv")

# Independent & dependent variables
x = dataset.iloc[:, :-1]   # Features
y = dataset.iloc[:, -1]    # Target

# Fill missing numerical values
imputer = SimpleImputer()
x.iloc[:, :] = imputer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

# Plot
plt.scatter(x_test.iloc[:, 0], y_test, color='red')
plt.plot(x_train.iloc[:, 0], regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Coefficients
m = regressor.coef_
c = regressor.intercept_
yr_12 = m[0] * 12 + c
print(f"Predicted salary for 12 years experience: {yr_12}")


bias = regressor.score(x_train,y_train)
bias
variance = regressor.score(x_test,y_test)
variance