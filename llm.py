
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time
import os
import ntpath
import json
from langchain.docstore.document import Document


load_dotenv()
groq_api_key = os.getenv("Groq_API_KE")
cohere_api_key = os.getenv("Cohere_API_KEY")

directory_path = os.path.dirname(os.path.abspath("__file__"))

class LLM:
    def __init__(self):        
        self.llm = ChatGroq(api_key=groq_api_key, temperature=0, model_name="llama-3.3-70b-specdec")
        self.embedding_model = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
        
        system = """You are a helpful assistant who answers only based on its context
                    context : {context}"""

        human = """{question}"""
        self.prompt = ChatPromptTemplate.from_messages([("system", system), 
                                                        ("human", human)])
        self.filter = None
        if not os.path.exists("log.json"):
            self.log = {"EmbeddedDocsName":{}}
            self.save_log()
        else:
            self.load_log()
            
        self.final_faiss_db = None
    def save_log(self):
        with open("log.json", "w") as file:
            file.write(json.dumps(self.log))
    def load_log(self):
        with open("log.json", "r") as file:
            self.log = json.loads(file.read())
    def get_model_prediction(self, resume):
        chain = self.prompt | self.chat | StrOutputParser()
        ai_message = chain.invoke(resume)
        return ai_message.content
    
    def load_docs(self, selected_docs_name, cluster_name):
        file_paths = [os.path.join("temp", cluster_name, f_name) for f_name in selected_docs_name if f_name not in self.log["EmbeddedDocsName"].get(cluster_name, [])]
        self.documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                pdf_loader = PyPDFLoader(file_path)
                docs = pdf_loader.load()
            elif file_path.endswith(".txt"):
                txt_loader = TextLoader(file_path)
                docs = txt_loader.load()
            file_name = ntpath.basename(file_path)
            docs_with_metadata = [Document(page_content=doc.page_content,  metadata={"source": file_name}) 
                                  for doc in docs]
            self.documents.extend(docs_with_metadata)
            if cluster_name not in self.log["EmbeddedDocsName"]:
                self.log["EmbeddedDocsName"][cluster_name] = [file_name]
            else:
                self.log["EmbeddedDocsName"][cluster_name].append(file_name)
                
        
    def update_rag(self):
        self.rag = RunnableParallel(
            {
            "context": itemgetter("question") | self.retriever,
            "question": RunnablePassthrough(),
            }
            )
    
    def get_model_prediction_with_rag(self, question):
        chain = self.rag | self.prompt | self.llm.with_config(temperature=0.1) | StrOutputParser()
        ai_message = chain.invoke({"question":question})
        return ai_message
    
    def split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        for text_chunk in text_chunks:
            lines = text_chunk.page_content.splitlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            text_chunk.page_content = "\n".join(cleaned_lines)
        return text_chunks
    
    def split_chunks_to_max_limited(self, text_chunks):
        n_tokens = 0
        i_0 = 0
        splitted_chunks = []
        for i, chunk in enumerate(text_chunks):
            print(chunk.metadata)
            n_tokens += len(chunk.page_content.split())
            if n_tokens > 10000:
                splitted_chunks.append(text_chunks[i_0: i])
                i_0 = i
                n_tokens = len(chunk.page_content.split())
            if len(text_chunks) == i + 1 :
                splitted_chunks.append(text_chunks[i_0: ])
        return splitted_chunks
    
    def build_db(self, splitted_chunks):
        faiss_stores = []
        for i, batch in enumerate(splitted_chunks):
            db = FAISS.from_documents(batch, self.embedding_model)
            faiss_stores.append(db)
            if len(splitted_chunks) == i + 1:
                print("Embedding Done ....")
                break
            
            print("sleeping for 70 seconds")
            time.sleep(70)
        self.final_faiss_db = faiss_stores[0]
        for db in faiss_stores[1:]:
            self.final_faiss_db.merge_from(db)
        return self.final_faiss_db
    
    def save_db(self, cluster_name):
        self.final_faiss_db.save_local(os.path.join(directory_path, "faiss_dbs", cluster_name))
    def load_db(self, cluster_name):
        db_path = os.path.join(directory_path, "faiss_dbs", cluster_name)
        loaded_db = FAISS.load_local(db_path, self.embedding_model, allow_dangerous_deserialization=True)
        if self.final_faiss_db:
            self.final_faiss_db.merge_from(loaded_db)
        else:
            self.final_faiss_db = loaded_db

    def get_retriever(self, filter=None):
        self.retriever = self.final_faiss_db.as_retriever(search_kwargs={"k": 5, "filter":{"source":filter}})
    
    def process_new_docs(self, new_docs_name, cluster_name):
        self.load_docs(new_docs_name, cluster_name)
        text_chunks = self.split_documents(self.documents)
        splitted_chunks = self.split_chunks_to_max_limited(text_chunks)
        self.build_db(splitted_chunks)
        
    def run(self, question, cluster_name, selected_docs_name):
        new_docs_name = set(selected_docs_name) - set(self.log["EmbeddedDocsName"].get(cluster_name, []))
        common_docs_name = set(selected_docs_name).intersection(self.log["EmbeddedDocsName"].get(cluster_name, []))
        if new_docs_name : 
            print("new docs ... processing ....")
            self.process_new_docs(new_docs_name, cluster_name)
        if common_docs_name:
            print("old docs .... loading ...")
            self.load_db(cluster_name)
        self.save_db(cluster_name)
        print("DB has saved ... ")
        self.get_retriever(selected_docs_name)
        self.update_rag()
        ai_ans = self.get_model_prediction_with_rag(question=question)
        self.save_log()
        print("AI message has received successfully ...")
        print(self.log)
        
        return ai_ans
    
    
        
        
