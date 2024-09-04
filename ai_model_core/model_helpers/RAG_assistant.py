# model_helpers/RAG_assistant.py
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

class RAGAssistant:
    def __init__(self, model_name="mistral", embedding_model="nomic-embed-text"):
        self.model_local = ChatOllama(model=model_name)
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.retriever = None

    def load_and_process_documents(self, sources):
        # 1. Load documents
        docs = []
        for source in sources:
            if source.startswith("http"):
                loader = WebBaseLoader(source)
            elif source.endswith(".pdf"):
                loader = PyPDFLoader(source)
            else:
                raise ValueError(f"Unsupported source type: {source}")
            docs.extend(loader.load())

        # 2. Split documents
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs)

        # 3. Create embeddings and store in vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(base_url='http://127.0.0.1:11434', model=self.embedding_model),
        )
        self.retriever = self.vectorstore.as_retriever()

    def setup_rag_chain(self):
        # 4, 5, 6, 7. Set up RAG chain
        rag_prompt_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
        
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.model_local
            | StrOutputParser()
        )

    def query(self, question):
        if not self.rag_chain:
            raise ValueError("RAG chain not set up. Call setup_rag_chain() first.")
        return self.rag_chain.invoke(question)

# Usage example
if __name__ == "__main__":
    assistant = RAGAssistant()
    
    # Load and process documents
    sources = [
        "https://ollama.com/",
        "https://ollama.com/blog/windows-preview",
        "https://ollama.com/blog/openai-compatibility",
        # Add more sources as needed
    ]
    assistant.load_and_process_documents(sources)
    
    # Set up the RAG chain
    assistant.setup_rag_chain()
    
    # Example query
    question = "What is Ollama?"
    answer = assistant.query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")