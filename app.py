import boto3
import streamlit as st
import re
import pandas as pd
## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

# Function to remove URLs, mentions, emojis, and hashtags from text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove emojis by filtering out characters outside of basic ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    text = text.replace('\n', ' ')
    return text

def clean_posts():
    excel_file_path = 'data/AS Social Media Posts.xlsx'
    sheets_info = {
        'FB': {'column_name': 'Description', 'header_row': 1},
        'TW': {'column_name': 'Tweet text', 'header_row': 1},
        'IG': {'column_name': 'Description', 'header_row': 0},
    }

    cleaned_posts = []  # List to store cleaned posts for vectorization

    for sheet_name, info in sheets_info.items():
        # Adjust header based on the sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=info['header_row'])

        column_name = info['column_name']
        # Remove duplicate and missing posts
        df.drop_duplicates(subset=[column_name], inplace=True)
        df.dropna(subset=[column_name], inplace=True)

        # Clean post texts
        df[column_name] = df[column_name].apply(clean_text)

        # Append cleaned posts to the list
        cleaned_posts.extend(df[column_name].tolist())
        
    # `cleaned_posts` now contains all cleaned posts ready for further processing
    df = pd.DataFrame(cleaned_posts, columns = ['text'])
    df.to_csv('data/merged_posts.csv')

## Data ingestion
def data_ingestion():
    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    clean_posts()

    pdfLoader=PyPDFDirectoryLoader("data")
    pdfDocuments=pdfLoader.load()

    csvLoader = CSVLoader("data/merged_posts.csv")
    csvDocuments= csvLoader.load()

    pdfDocs=text_splitter.split_documents(pdfDocuments)
    csvDocs=text_splitter.split_documents(csvDocuments)
    docs = pdfDocs + csvDocs
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
# Replicate Credentials
def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question. If you don't know the answer, 
try it again, rethink about the answer and provide the answer in maximum 250 words.
Don't make the answer and always provide factual information.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

# App title
st.set_page_config(page_title="💬 Sian Chatbot")


with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,  allow_dangerous_deserialization = True)
            llm=get_llama2_llm()
            
            #faiss_index = get_vector_store(docs)
            response = get_response_llm(llm,faiss_index,prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)