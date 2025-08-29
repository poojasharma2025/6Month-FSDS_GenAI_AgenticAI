import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r'C:\Users\WELCOME\Desktop\Python\linear_regression\house_price_model.pkl','rb'))


# Set the title of the Streamlit app
st.title("🏡 HOUSE PRICE PREDICTION APP")


# Add a brief description
st.write("This app predicts the price based on living house 🏡 square feet using a simple linear regression model.")

# Add input widget for user to enter years of experience
price_sqrft = st.number_input("ENTER PRICE($) OF PER SQRFT:", min_value=0.0, max_value=600000.0, value=1.0, step=1000.0)

# When the button is clicked, make predictions
if st.button("Predict House Price"):
    # Make a prediction using the trained model
    sqrft_input = np.array([[price_sqrft]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(sqrft_input)
   
    # Display the result
    st.success(f"The predicted  Price for {price_sqrft}square feet is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of House data 🏡 and Price of square feet.and built model by Pooja sharma")