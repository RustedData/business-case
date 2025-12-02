import streamlit as st

st.title("Streamlit App Mock")
st.header("Welcome to the Streamlit Cloud Deployment Mock")

st.write("This is a basic Streamlit app ready for deployment to Streamlit Cloud.")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Hello, {name}! Welcome to the app.")

st.button("Click me!")