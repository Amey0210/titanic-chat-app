import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Titanic AI Agent", page_icon="🚢")

st.title("🚢 Titanic Dataset Chat Agent")
st.markdown("Exploring passenger data using **Llama 3.3** on Groq.")

# --- DATA & AGENT SETUP ---
@st.cache_data
def load_data():
    return pd.read_csv("data/titanic.csv")

df = load_data()

# Load API Key from Streamlit Secrets (for Cloud) or Environment (for Local)
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets or a .env file.")
    st.stop()

llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0)

agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(base64.b64decode(msg["image"]))

if prompt := st.chat_input("Ask about the Titanic passengers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            try:
                export_path = "temp_chart.png"
                if os.path.exists(export_path): os.remove(export_path)
                
                # We tell the agent to save charts locally in the cloud container
                full_query = f"{prompt}. If a chart is requested, save it as '{export_path}' using matplotlib."
                
                response = agent.invoke({"input": full_query})
                answer = response.get("output", str(response))
                
                st.markdown(answer)
                
                img_str = None
                if os.path.exists(export_path):
                    with open(export_path, "rb") as f:
                        img_str = base64.b64encode(f.read()).decode()
                    st.image(base64.b64decode(img_str))
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "image": img_str})
                
            except Exception as e:
                st.error(f"Error: {e}")