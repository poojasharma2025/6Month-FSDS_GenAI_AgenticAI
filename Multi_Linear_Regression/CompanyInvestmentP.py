import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\WELCOME\Documents\Data Science\August 2025\20TH Aug-MLR\20th - mlr\MLR\Investment.csv")

# One-hot encode categorical column 'State'
df = pd.get_dummies(df, columns=["State"], drop_first=True)

# Features and target
X = df.drop("Profit", axis=1)
y = df["Profit"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("Digital_marketing_model1.pkl", "wb"))
print("✅ Model retrained and saved as Digital_marketing_model1.pkl")