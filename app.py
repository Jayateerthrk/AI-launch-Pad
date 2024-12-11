
import streamlit as st
import os
import openai
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import tempfile  # To handle file saving for uploaded PDFs
import ast  # To parse string representation of dict

# Streamlit UI elements for User Profile and Links
#st.sidebar.image("https://your-image-url.com/Jay_Image.jpeg", width=150)  # Replace with your image URL
st.sidebar.markdown("## [Jayateerth Katti](https://www.linkedin.com/in/jayateerth-katti-10103a14/)")
st.sidebar.markdown("Manager-Testing")
st.sidebar.markdown("""
A seasoned Test Manager and AI enthusiast. I am here to solve your testing problems.
""")

# Title for the app
st.title("QA Assistant")

# OpenAI API key input
OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# File uploader for PDF
uploaded_pdf = st.file_uploader("Upload the Requirements PDF", type="pdf")

# Display options
pd.set_option("display.max_colwidth", None)

# LLM_NAME should be a valid model name
LLM_NAME = "gpt-4o-mini"

# Updated Prompt Template to reflect the new use case
PROMPT_TEMPLATE = """You are a QA Assistant, a helpful AI assistant skilled in reviewing and analyzing requirements.
Your task is to assist the user by answering questions related to the project requirements.
You will be given a question and relevant excerpts from the provided Requirements Document.
Please provide short and clear answers based on the provided context. Additionally, if relevant, provide follow-up questions that a QA might ask to clarify the requirements further.

Context:
{context}

Question:
{question}

Your detailed answer:
"""

# Function to generate context storage from PDF chunks
def get_context_storage(pdf_file_path) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    docs = PyPDFLoader(pdf_file_path).load_and_split(text_splitter)
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    return db

# Main application logic
if uploaded_pdf:
    st.write("Processing the uploaded PDF...")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(uploaded_pdf.read())  # Save the uploaded file to disk
        temp_pdf_path = temp_pdf_file.name  # Get the path of the temporary file

    # Create the chain
    llm = ChatOpenAI(model=LLM_NAME, temperature=0)  # Use ChatOpenAI for chat models
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=get_context_storage(temp_pdf_path).as_retriever(), prompt=prompt)

    # Input for user's question
    question = st.text_input("Ask a question about the requirements:")

    # Display answer
    if question:
        answer = qa_chain(question)

        # Check if answer is a string and parse it to a dictionary
        if isinstance(answer, str):
            try:
                answer_dict = ast.literal_eval(answer)
            except:
                answer_dict = {}
        else:
            answer_dict = answer

        # Extract 'result' and 'follow-up questions' from the answer
        result_text = answer_dict.get('result', 'No result found.')
        follow_up_text = answer_dict.get('follow-up questions a QA might ask:', 'No follow-up questions.')

        # Display the result with proper formatting
        st.markdown("### **Answer:**")
        st.markdown(result_text)

        # Display follow-up questions with bullet points
        st.markdown("### **Follow-up Questions:**")

        # Split the follow-up questions by newline and format as a list
        follow_up_questions = follow_up_text.strip().split('\n')
        for q in follow_up_questions:
            # Remove leading hyphens and whitespace
            formatted_question = q.lstrip('- ').strip()
            if formatted_question:
                st.markdown(f"- {formatted_question}")

    # Clean up the temporary file
    os.remove(temp_pdf_path)

# Footer for App
st.markdown("""
---
*This QA Assistant App is built by [Jayateerth Katti](https://www.linkedin.com/in/jayateerth-katti-10103a14/).*
Connect with me for more insights.
""")
