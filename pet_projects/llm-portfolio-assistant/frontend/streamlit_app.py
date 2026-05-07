# streamlit_app.py
"""
Streamlit Chat Interface for RAG Portfolio API

This app provides a clean chat interface to interact with the RAG API.
Users can ask questions about the ML portfolio and get AI-generated answers
based on semantic search over project documentation.
"""

import streamlit as st
import requests

# Page configuration - sets browser tab title and icon
st.set_page_config(page_title="Portfolio Assistant", page_icon="🤖")

# Main title and description
st.title("🤖 Portfolio Assistant")
st.markdown("Ask me anything about my projects!")

# ============================================================================
# SESSION STATE (Chat History)
# ============================================================================
# session_state persists across reruns and stores conversation history
# Initialize empty message list if not exists (first run)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================================
# DISPLAY CHAT HISTORY
# ============================================================================
# Loop through all previous messages and display them
# Each message has a "role" (user or assistant) and "content" (text)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):      # Creates styled message bubble
        st.markdown(msg["content"])          # Renders the message text

# ============================================================================
# HANDLE USER INPUT
# ============================================================================
# chat_input() returns the entered text when user presses Enter
# prompt = None when no input, otherwise contains the question

if prompt := st.chat_input("Ask about churn prediction, traffic signs, etc."):
    
    # 1. Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Get AI response from backend API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):      # Show loading animation
            
            # Call the RAG API endpoint
            response = requests.post(
                "http://localhost:8000/ask",      # FastAPI endpoint
                json={"question": prompt}          # Request body
            )
            
            # Extract answer from JSON response
            answer = response.json()["answer"]
            st.markdown(answer)                     # Display the answer
    
    # 3. Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})