import streamlit as st
import joblib

# Load model
model = joblib.load("spam_model.pkl")
cv = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection System")

message = st.text_area("Enter Email Message")

if st.button("Check"):
    data = cv.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚫 Spam Email")
    else:
        st.success("✅ Not Spam")