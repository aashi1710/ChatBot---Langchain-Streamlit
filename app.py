import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Safety check for API key
if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


st.title("Hi, How can I help you today?")


query = st.text_input("Search Tab", placeholder="Type your question here...")

# Create Gemini LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key  # Pass API key directly
)


if st.button("Search"):
    if query.strip():
        
            response = llm.invoke(query)
            st.markdown(f"""
            <div style='border:2px solid #ddd; border-radius:10px; padding:15px;'>
                {response.content}
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("Please type a question first.")
