import streamlit as st
import os
from llm import LLM
st.set_page_config(page_title="AI Document Assistant", layout="wide")

# Define base directory for storage
BASE_DIR = "temp"
llm_model = LLM()

# Ensure the directory exists
os.makedirs(BASE_DIR, exist_ok=True)

# Function to save uploaded file to a specific cluster
def save_file(uploaded_file, cluster_name):
    cluster_path = os.path.join(BASE_DIR, cluster_name)
    os.makedirs(cluster_path, exist_ok=True)  # Create cluster directory if not exists

    file_path = os.path.join(cluster_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# Load existing clusters
existing_clusters = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

# Let users select or create a cluster
cluster_name = st.text_input("Enter cluster name (or select existing one):")

if existing_clusters:
    selected_cluster = st.selectbox("Or select an existing cluster:", existing_clusters)
    if selected_cluster:
        cluster_name = selected_cluster

if cluster_name:
    st.write(f"Selected Cluster: {cluster_name}")

    # Upload files
    uploaded_files = st.file_uploader("Upload Documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_file(uploaded_file, cluster_name)
        st.success("Files uploaded successfully!")

    # List files in selected cluster
    cluster_path = os.path.join(BASE_DIR, cluster_name)
    if os.path.exists(cluster_path):
        docs_names = os.listdir(cluster_path)
    else:
        docs_names = []

    if docs_names:
        selected_docs = st.multiselect("Select Documents", options=docs_names)

        if selected_docs:
            st.write("You selected:", " ,".join(selected_docs))

            # User input for question
            question = st.text_input("Ask a question about the selected documents:")

            if st.button("Submit"):
                st.write(f"You asked: {question}")
                ai_ans = llm_model.run(question, cluster_name, selected_docs)
                st.write(f"Processing... {ai_ans}")
