# model_helpers/RAG_assistant.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from operator import add
#import BeautifulSoup4
import os
import pypdf
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ai_model_core import get_model, get_embedding_model, get_prompt_template, get_system_prompt, _format_history
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
class State(TypedDict):
    input: str
    context: List[str]
    question: str
    answer: str
    all_actions: Annotated[List[str], add]
class RAGAssistant:
    def __init__(self, model_name="Ollama (LLama3.1)", embedding_model="nomic-embed-text", chunk_size=7500, chunk_overlap=100, temperature=0.7, num_similar_docs=3, language="english"):
        self.model_local = get_model(model_name)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.num_similar_docs = num_similar_docs
        self.vectorstore = None
        self.retriever = None
        self.graph = StateGraph(State)
        self.language = language
        self.prompt_template = None
        self.use_history = True
        self.config = load_config()
    
    def load_content(self, url_input, file_input, model_choice, embedding_choice, chunk_size, chunk_overlap):
        try:
            # Update the current instance instead of creating a new one
            self.model_local = get_model(model_choice)
            self.embedding_model_name = embedding_choice
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
            # Call the setup_vectorstore method
            self.setup_vectorstore(url_input, file_input)
            
            return "Content loaded successfully into the vectorstore."
        except Exception as e:
            return f"Error loading content: {str(e)}"
        
    def setup_vectorstore(self, urls, files):
        docs = []
        # Check and process URLs if they are provided
        if urls and isinstance(urls, str):
            print(f"URLs: {urls}")
            urls_list = urls.split("\n")
            for url in urls_list:
                url = url.strip()
                if url:
                    print(f"Loading URL: {url}")
                    try:
                        loaded_docs = WebBaseLoader(url).load()
                        if loaded_docs:
                            docs.extend(loaded_docs)
                        else:
                            print(f"Warning: URL {url} returned no content.")
                    except Exception as e:
                        print(f"Error loading URL {url}: {str(e)}")
        else:
            print("No valid URLs provided.")
            
        # Process uploaded files
        if files:
            print(f"Files: {files}")
            for file in files:
                file_extension = os.path.splitext(file.name)[1].lower()
                if file_extension == '.txt':
                    docs.extend(TextLoader(file.name).load())
                elif file_extension == '.pdf':
                    docs.extend(PyPDFLoader(file.name).load())
                elif file_extension == '.docx':
                    docs.extend(Docx2txtLoader(file.name).load())
        else:
            print("No files provided.")
                
        if docs:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            doc_splits = text_splitter.split_documents(docs)
        else:
            raise ValueError("No documents were loaded.")

        embedding_model = get_embedding_model(self.embedding_model_name)

        self.vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.num_similar_docs})

    def retrieve_context(self, state):
        question = state['question']
        context = self.retriever.get_relevant_documents(question)
        return {"context": [doc.page_content for doc in context]}

    def generate_answer(self, state):
        context = state['context']
        question = state['question']
        
        rag_prompt_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
        
        chain = (
            rag_prompt 
            | self.model_local.bind(temperature=self.temperature)
            | StrOutputParser()
        )
        
        answer = chain.invoke({"context": context, "question": question})
        return {"answer": answer, "all_actions": ["answer_generated"]}

    def setup_graph(self):
        self.graph.add_node("retrieve_context", self.retrieve_context)
        self.graph.add_node("generate_answer", self.generate_answer)

        self.graph.add_edge("retrieve_context", "generate_answer")
        self.graph.add_edge("generate_answer", END)

        self.graph_runnable = self.graph.compile()

async def query(self, question, history=None, prompt_template=None):
    if not self.vectorstore or not self.retriever:
        raise ValueError("Vector store or retriever not set up. Call setup_vectorstore() first.")
    
    # Retrieve relevant documents
    relevant_docs = self.retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    # Base RAG prompt
    base_rag_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
    base_rag_prompt = ChatPromptTemplate.from_template(base_rag_template)
    
    # Construct the chain
    if prompt_template:
        # If a custom prompt template is set, use it to format the question
        custom_prompt = get_prompt_template(prompt_template, self.config)
        rag_chain = (
            {"context": lambda _: context, "question": custom_prompt}
            | base_rag_prompt
            | self.model_local
            | StrOutputParser()
        )
    else:
        # If no custom prompt template, use the base RAG prompt directly
        rag_chain = (
            base_rag_prompt
            | self.model_local
            | StrOutputParser()
        )
    
    input_dict = {"question": question, "context": context}
    if self.use_history and history:
        input_dict["history"] = _format_history(history)
    
    response = await rag_chain.ainvoke(input_dict)
    return response