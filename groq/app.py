import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

import time
from dotenv import load_dotenv

load_dotenv()

# load the groq api key

groq_api_key = os.environ['GROQ_API_KEY']


if "vector" not in st.session_state:
    st.session_state.embedding = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)

st.title("Groq Demo")
llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
               )

prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provied context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
                                                
        Question : {input}
        """)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input you promt here")

if prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input" : prompt})
    print("Response Time : ", time.process_time() - start)
    st.write(response['answer'])


    # with streamlit expander
    with st.expander("Document samilarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------------------------")



