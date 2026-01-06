import streamlit as st

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Real-Time Sign Language Detection App")
st.write("Welcome to your sign language interface!")

st.markdown("""
### Navigation  
Use the menu on the left to move across:

- **Live Detection**
- **Add New Sign**
- **Train Model**
- **Settings**
""")
