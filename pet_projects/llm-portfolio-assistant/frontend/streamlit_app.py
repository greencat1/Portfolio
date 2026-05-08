# streamlit_app.py
"""
Streamlit Chat Interface for RAG Portfolio API
"""

import streamlit as st
import requests
import os



API_URL = os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(page_title="Portfolio Assistant", page_icon="🤖")

st.title("🤖 Portfolio Assistant")
st.markdown("Ask me anything about my projects!")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about churn prediction, traffic signs, etc."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt},
                    timeout=60
                )
                answer = response.json()["answer"]
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                answer = f"Sorry, I couldn't connect to the API. Error: {e}"
                st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})