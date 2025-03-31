import streamlit as st
import os
from llm import LLM

llm_model = LLM()
# ---- Streamlit UI ----
st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI-Powered Document Assistant")
import streamlit as st
import os
from io import BytesIO
from PyPDF2 import PdfReader
import docx
selected_docs = []

directory_path = os.path.dirname(os.path.abspath("__file__"))
os.makedirs("temp", exist_ok=True)

def save_file(file):
    temp_path = os.path.join("temp", file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
        
# Streamlit UI
st.title("Upload and Select Documents")

# Upload files
uploaded_files = st.file_uploader("Upload Documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
existence_files_name = os.listdir("temp")
docs_names = existence_files_name
if uploaded_files:
    for uploaded_file in uploaded_files:
        docs_names.append(uploaded_file.name)
        save_file(uploaded_file)

if docs_names:
    selected_docs = st.multiselect("Select Documents", options=docs_names)

if selected_docs:
    print(selected_docs)
    # st.write("Selected Documents Content:")
    # for doc in selected_docs:
    #     st.write(f"**{doc}**")
    #     st.text_area("", doc_contents[doc], height=200)

    # User input for question
    question = st.text_input("Ask a question about the selected documents:")

    if st.button("Submit"):
        st.write(f"You asked: {question}")
        ai_ans = llm_model.run(question, selected_docs)
        st.write(f"Processing... {ai_ans}")
        


