# model_helpers/RAG_assistant.py
# Standard library imports
import asyncio
from typing import TypedDict, List, Annotated, Union
from operator import add

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModel
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ai_model_core import get_model, get_embedding_model, get_prompt_template, _format_history
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader

class State(TypedDict):
    input: str
    context: List[str]
    question: str
    answer: str
    all_actions: Annotated[List[str], add]

import fitz  # PyMuPDF
from langchain.schema import Document
class PyMuPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        docs = []
        doc = fitz.open(self.file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # Try to extract text directly
            
            if not text.strip():  # If no text was extracted, try OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": self.file_path
                    }
                ))
            else:
                print(f"Warning: No text extracted from page {page_num + 1}")
        
        return docs

class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class RAGAssistant:
    def __init__(self, model_name="Ollama (LLama3.1)", embedding_model="nomic-embed-text", retrieval_method="similarity", chunk_size=500, chunk_overlap=50, temperature=0.4, num_similar_docs=3, language="english", max_tokens=None):
        self.model_local = get_model(model_name)
        self.embedding_model_name = embedding_model
        self.embedding_model = get_embedding_model(embedding_model)  # Initialize the embedding model
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
        self.max_tokens = max_tokens
        self.content_loader = EnhancedContentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retrieval_method = retrieval_method
    
    def load_content(self, url_input: str, file_input: Union[str, List[str]]):
        try:
            docs = self.content_loader.load_and_split_document(file_input, url_input)
            self.setup_vectorstore(docs)
            return "Content loaded successfully into memory."
        except Exception as e:
            return f"Error loading content: {str(e)}"
        
    def setup_vectorstore(self, docs: List[Document]):
        if not docs:
            raise ValueError("No documents were loaded.")

        # If using the custom embedding function for all-MiniLM-L6-v2
        if isinstance(self.embedding_model, CustomHuggingFaceEmbeddings):
            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_model.embed_documents(texts)
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embedding_model,
            )
        else:
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embedding_model,
            )

        self.retriever = self.select_retriever(self.retrieval_method)

    def select_retriever(self, method):
        if method == "similarity":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.num_similar_docs}
            )
        elif method == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.num_similar_docs, "fetch_k": 20}
            )
        elif method == "similarity_threshold":
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8, "k": self.num_similar_docs}
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    async def retrieve_context(self, query: str) -> List[Document]:
        # Use run_in_executor to run the synchronous invoke method in a separate thread
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, self.retriever.invoke, query)
        return docs

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
            | self.model_local.bind(temperature=self.temperature, max_tokens=self.max_tokens)
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
        relevant_docs = await self.retrieve_context(question)
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
                {"context": RunnablePassthrough(), "question": custom_prompt}
                | base_rag_prompt
                | self.model_local.bind(temperature=self.temperature, max_tokens=self.max_tokens)
                | StrOutputParser()
            )
        else:
            # If no custom prompt template, use the base RAG prompt directly
            rag_chain = (
                base_rag_prompt
                | self.model_local.bind(temperature=self.temperature, max_tokens=self.max_tokens)
                | StrOutputParser()
            )
        
        input_dict = {"question": question, "context": context}
        if self.use_history and history:
            input_dict["history"] = _format_history(history)
        
        response = await rag_chain.ainvoke(input_dict)
        
        # Check if response is None and handle it
        if response is None:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question or providing more context."
        
        return response