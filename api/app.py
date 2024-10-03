from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI
from langchain_community.llms import Ollama

import streamlit as st
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()


## environment variables call
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A chatbot api server"
)


add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)


## creating chatbot

# open ai
model = ChatOpenAI()

# llama2
llm = Ollama(model="llama3.2:3b")

prompt = ChatPromptTemplate.from_template("Write me an essay {topic} with in 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem {topic} for 5 years old child with in 100 words")


## chain
add_routes(
    app,
    prompt|model,
    path="/essay"
)


add_routes(
    app,
    prompt2|llm,
    path="/poem"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)