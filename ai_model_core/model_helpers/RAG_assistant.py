# model_helpers/RAG_assistant.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from operator import add
#import BeautifulSoup4
import os
import pypdf
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
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
        
        # Process URLs
        for url in urls_list:
            url = url.strip()
            if url:
                loaded_docs = WebBaseLoader(url).load()
                if loaded_docs is not None:
                    docs.extend(loaded_docs)
                else:
                    print(f"Failed to load content from: {url}")
        
        # Process uploaded files
        for file in files:
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension == '.txt':
                docs.extend(TextLoader(file.name).load())
            elif file_extension == '.pdf':
                docs.extend(PyPDFLoader(file.name).load())
            elif file_extension == '.docx':
                docs.extend(UnstructuredWordDocumentLoader(file.name).load())
        
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        doc_splits = text_splitter.split_documents(docs)

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

    async def query(self, question, history=None):
        if not self.vectorstore or not self.retriever:
            raise ValueError("Vector store or retriever not set up. Call setup_vectorstore() first.")
        
        if not self.prompt_template:
            raise ValueError("Prompt template not set. Set the prompt_template attribute first.")
        
        prompt_template = get_prompt_template(self.prompt_template, self.config)
        
        rag_chain = (
            {
                "context": self.retriever,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | self.model_local
            | StrOutputParser()
        )
        
        input_dict = {"question": question}
        if self.use_history and history:
            input_dict["history"] = _format_history(history)
        
        response = await rag_chain.ainvoke(input_dict)
        return response