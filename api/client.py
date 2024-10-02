import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/eassay/invoke",
                             json={"input": {"topic":input_text}})
    return response.json()['output']['content']

def get_llama2_response(input_text2):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={"input": {"topic":input_text2}})
    return response.json()['output']
# streamlit framework

st.title("Langchain Demo ChatBot openai llam2 API Chains")
input_text = st.text_input("write a essay on")
input_text2 = st.text_input("write a poem on")

if input_text:
    st.write(get_openai_response(input_text))

if input_text2:
    st.write(get_openai_response(input_text2))