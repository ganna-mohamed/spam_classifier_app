# app.py

import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App UI
st.title("📧 Spam Email Classifier")
st.write("Enter a message below to check if it's Spam or Not Spam:")

# User input
user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.error("🚨 This is a SPAM message!")
        else:
            st.success("✅ This is NOT SPAM.")