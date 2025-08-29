import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open(r"C:\Users\WELCOME\Desktop\Python\Multi_Linear_Regression\Digital_marketing_model.pkl", "rb"))

# Title
st.title("📊 Business Investment Profit Predictor")

# Sidebar for navigation
st.sidebar.header("User Input Features")

# Input fields
digital = st.sidebar.number_input("💻 Digital Marketing Investment ($)", min_value=0.0, step=1000.0)
promotion = st.sidebar.number_input("📢 Promotion Investment ($)", min_value=0.0, step=1000.0)
research = st.sidebar.number_input("🔬 Research Investment ($)", min_value=0.0, step=1000.0)

# Dropdown for State
state = st.sidebar.selectbox("🏙️ Select State", ("Hyderabad", "Bangalore", "Chennai"))

# Convert state into one-hot encoding (like training)
state_map = {
    "Hyderabad": [1, 0, 0],
    "Bangalore": [0, 1, 0],
    "Chennai": [0, 0, 1]
}
state_encoded = state_map[state]

# Predict button
if st.sidebar.button("🚀 Predict Profit"):
    # Create input array (must match training format)
    input_data = np.array([[digital]])
    prediction = model.predict(input_data)
    
    # Show result
    st.success(f"✅ Predicted Profit: ${prediction[0]:,.2f}")
    # Display information about the model
st.write("The model was trained using a dataset of Investment and Profit of money.built model by Pooja sharma")