
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
import time
import os, sys
import ntpath
import json
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore

directory_path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(directory_path, ".env"))
groq_api_key = os.getenv("Groq_API_KEY")
aval_ai_api_key = os.getenv("Aval_AI_API_KEY")

class LLM:
    def __init__(self, user_id="default"):        
        self.llm = ChatGroq(api_key=groq_api_key, temperature=0.1, model_name="llama-3.1-8b-instant")
        # self.llm = OpenAI(api_key=aval_ai_api_key, temperature=0.1, model_name="gemini-2.0-flash", base_url="https://api.avalai.ir/v1")
        self.embedding_model = OpenAIEmbeddings(api_key=aval_ai_api_key, model="text-embedding-3-small", 
                                                base_url="https://api.avalai.ir/v1",
                                                timeout=30)
        system = """You are an intelligent assistant in a Telegram bot that helps users understand their uploaded documents.
                    Only answer questions using the information provided in the context below (the context may be in English or Persian).
                    ✅ Always give clear, accurate, and detailed answers that directly use the information in the context.  
                    🗣️ Always reply in the same language the user uses in their question (English or Persian).  
                    ---
                    context:
                    {context}
                    """

        # system = """You are a helpful customer support assistant for a smart greenhouse system.
        #             You only answer based on the provided context (which may be in English or Persian).
        #             You must give clear, practical, and accurate responses related to greenhouse monitoring, control, climate, sensors, or crops.
        #             Do NOT use vague or meaningless words.
        #             Do NOT use any of your previous answers as context.
        #             If you're not sure about something or it's outside the context, politely inform the user.
        #             Always reply in the same language the user uses.

        #             context: {context}"""

        human = """User question: {question}"""
        self.prompt = ChatPromptTemplate.from_messages([("system", system),
                                                        MessagesPlaceholder(variable_name="chat_history"),
                                                        ("human", human)])
        self.filter = None
        self.user_id = user_id
        self.his_messages = []
        self.log_dir = os.path.join(directory_path, "log", self.user_id)
        if not os.path.exists(os.path.join(self.log_dir, "log.json")):
            os.makedirs(self.log_dir, exist_ok=True)
            self.log = {self.user_id: {"EmbeddedDocsName":{}}}
            self.save_log()
        else:
            self.load_log()
            
        self.final_faiss_db = None
        self.retriever = None
        
        self.previous_selected_docs_name = []
        self.previous_selected_cluster = None
    def save_log(self):
        with open(os.path.join(self.log_dir, "log.json"), "w") as file:
            file.write(json.dumps(self.log))
    def load_log(self):
        with open(os.path.join(self.log_dir, "log.json"), "r") as file:
            self.log = json.loads(file.read())
    def get_model_prediction(self, resume):
        chain = self.prompt | self.chat | StrOutputParser()
        ai_message = chain.invoke(resume)
        return ai_message.content
    
    def load_docs(self, selected_docs_name, cluster_name):
        file_paths = [os.path.join(directory_path, "temp", self.user_id, cluster_name, f_name) for f_name in selected_docs_name if f_name not
                      in self.log[self.user_id]["EmbeddedDocsName"].get(cluster_name, [])]
        self.documents = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                pdf_loader = PyPDFLoader(file_path)
                docs = pdf_loader.load()
            elif file_path.endswith(".txt"):
                txt_loader = TextLoader(file_path)
                docs = txt_loader.load()
            elif file_path.endswith(".docx"):
                docx_loader = Docx2txtLoader(file_path)
                docs = docx_loader.load()
                
            file_name = ntpath.basename(file_path)
            docs_with_metadata = [Document(page_content=doc.page_content,  metadata={"source": file_name}) 
                                  for doc in docs]
            self.documents.extend(docs_with_metadata)
            if cluster_name not in self.log[self.user_id]["EmbeddedDocsName"]:
                self.log[self.user_id]["EmbeddedDocsName"][cluster_name] = [file_name]
            else:
                self.log[self.user_id]["EmbeddedDocsName"][cluster_name].append(file_name)
                
        
    def update_rag(self):
        self.rag = RunnableParallel(
            {
            "context": itemgetter("question") | self.retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x : self.his_messages[-10:]
            }
            )
    
    def get_model_prediction_with_rag(self, question):
        # chain = self.rag | self.prompt | self.llm.with_config(temperature=0.1) | StrOutputParser()
        # ai_message = chain.invoke({"question":question})
        rag_result = self.rag.invoke({"question":question})  # or whatever input you pass
        prompt_result = self.prompt.invoke(rag_result)
        llm_result = self.llm.with_config(temperature=0.1).invoke(prompt_result)
        ai_message = StrOutputParser().invoke(llm_result)
        documents = rag_result.get("context", [])  # or whatever key your RAG returns
        # Extract text content from each document
        text_blocks = [doc.page_content for doc in documents]

        # Save to a UTF-8 text file
        with open(os.path.join(directory_path,"prompt_result.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_result.to_string())
        with open(os.path.join(directory_path,"rag_output.txt"), "w", encoding="utf-8") as f:
            for i, block in enumerate(text_blocks):
                f.write(f"--- Document {i+1} ---\n{block}\n\n")
        
        self.his_messages.append(HumanMessage(content=question))
        self.his_messages.append(AIMessage(content=ai_message))
        print(self.his_messages)
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
        if os.path.exists(os.path.join(directory_path, "faiss_dbs", self.user_id, cluster_name)):
            self.load_db(cluster_name)
        self.final_faiss_db.save_local(os.path.join(directory_path, "faiss_dbs", self.user_id, cluster_name))
    def load_db(self, cluster_name):
        db_path = os.path.join(directory_path, "faiss_dbs", self.user_id, cluster_name)
        loaded_db = FAISS.load_local(db_path, self.embedding_model, allow_dangerous_deserialization=True)
        if self.final_faiss_db:
            if set(self.final_faiss_db.index_to_docstore_id.values()).intersection(set(loaded_db.index_to_docstore_id.values())) == set():
                self.final_faiss_db.merge_from(loaded_db)
            else:
                print("the same db, can't merge ...")
        else:
            self.final_faiss_db = loaded_db

    def get_retriever(self, allowed_sources:list=None):
        print("allowed sources:",  allowed_sources)
        # self.retriever = self.final_faiss_db.as_retriever(search_kwargs={"k": 5, "filter":{"source":filter}})
        self.retriever = self.get_filtered_retriever(self.final_faiss_db, allowed_sources)
    
    def process_new_docs(self, new_docs_name, cluster_name):
        self.load_docs(new_docs_name, cluster_name)
        text_chunks = self.split_documents(self.documents)
        if text_chunks:
            splitted_chunks = self.split_chunks_to_max_limited(text_chunks)
            self.build_db(splitted_chunks)
            return True
        else:
            print("No Valid documents found ...")
            return False
        
    def load_or_save_doc(self, cluster_name, selected_docs_name):
        new_docs_name = set(selected_docs_name) - set(self.log[self.user_id]["EmbeddedDocsName"].get(cluster_name, []))
        common_docs_name = set(selected_docs_name).intersection(self.log[self.user_id]["EmbeddedDocsName"].get(cluster_name, []))
        if new_docs_name : 
            print("new docs ... processing ....")
            is_doc_valid = self.process_new_docs(new_docs_name, cluster_name)
            if is_doc_valid:
                self.save_db(cluster_name)
                print("DB has saved ... ")
                self.save_log()
                
        if common_docs_name:
            print("old docs .... loading ...")
            self.load_db(cluster_name)
        
        if not self.final_faiss_db:
            return False
        else:
            return True
        
    
    def run_chain(self, selected_docs_name: list, question: str, cluster: str):
        if type(self.retriever) == type(None) or not (self.previous_selected_docs_name == selected_docs_name) or (self.previous_selected_cluster != cluster):
            print("getting the retriever")
            self.get_retriever(selected_docs_name)
            self.previous_selected_docs_name = selected_docs_name
            self.previous_selected_cluster = cluster
            
        self.update_rag()
        ai_ans = self.get_model_prediction_with_rag(question=question)
        return ai_ans
    
    def get_filtered_retriever(self, all_faiss_db, allowed_sources: list):
        # Access all stored docs and the FAISS index
        original_docs = all_faiss_db.docstore._dict
        original_index = all_faiss_db.index

        filtered_docs = []
        filtered_embeddings = []

        for i, (doc_id, doc) in enumerate(original_docs.items()):
            if doc.metadata.get("source") in allowed_sources:
                embedding = original_index.reconstruct(i)
                filtered_docs.append(doc)
                filtered_embeddings.append(embedding)

        if not filtered_docs:
            raise ValueError("No documents matched the source filters!")

        # Build new FAISS index
        dimension = len(filtered_embeddings[0])
        new_index = IndexFlatL2(dimension)
        new_index.add(np.array(filtered_embeddings).astype("float32"))

        # Build docstore and ID map
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(filtered_docs)})
        index_to_docstore_id = {i: str(i) for i in range(len(filtered_docs))}

        # # Reconstruct FAISS object
        db = FAISS(
            index=new_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=self.embedding_model  # not called, just for querying
        )
        # db = FAISS.from_documents(filtered_docs, self.embedding_model)

        return db.as_retriever(search_kwargs={"k": 5})
    
    
        
        
