# Forked & Modified from - https://github.com/dataprofessor/langchain-ask-the-doc/blob/master/app-v1.py
import os
from dotenv import load_dotenv
import streamlit as st
# LLM Model
from langchain.llms import OpenAI
# Text Chunking
from langchain.text_splitter import CharacterTextSplitter
# Embedding Model
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# Vector database
from langchain.vectorstores import Chroma
# Chaining
from langchain.chains import RetrievalQA

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# QA function
def generate_response(uploaded_file, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create a vector store from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # LLM Model initialization
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='Document QA')
st.title('Document QA')

# File upload
uploaded_file = st.file_uploader('Upload a file', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'query description', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    # After Submission
    if submitted:
        with st.spinner('Calculating...'):
            # Run the QA function
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(response)
