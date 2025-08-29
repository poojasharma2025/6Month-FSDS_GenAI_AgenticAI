import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib

# Optional: set backend (TkAgg for interactive plots)
matplotlib.use("TkAgg")

np.set_printoptions(threshold=np.inf)

# Load the dataset
dataset = pd.read_csv(r'C:\Users\WELCOME\Documents\Data Science\August 2025\20TH Aug-MLR\20th - mlr\MLR\Investment.csv')

space = dataset['DigitalMarketing']
price = dataset['Profit']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Splitting the data into Train and Test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting simple linear regression to the Training Set
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predicting the prices
pred = regressor.predict(xtest)

# Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'green')
plt.plot(xtrain, regressor.predict(xtrain), color = 'black')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'orange')
plt.plot(xtrain, regressor.predict(xtrain), color = 'red')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Save the trained model to disk
filename = 'Digital_marketing_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

# Load the pickle file
with open('Digital_marketing_model.pkl','rb') as file:
    loaded_model = pickle.load(file)

print("Model has been pickled and saved as Digital_marketing_model.pkl")