# Standard library imports
import asyncio
from typing import TypedDict, List, Annotated, Union
from operator import add

# Third-party imports
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from typing import TypedDict, List, Annotated
from operator import add
#import BeautifulSoup4
import asyncio
import os
import pypdf
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ai_model_core import get_model, get_embedding_model, get_prompt_template, get_system_prompt, _format_history, load_documents, load_from_files, _load_from_urls, split_documents, load_and_split_document
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_core.utils import EnhancedContentLoader

class State(TypedDict):
    input: str
    context: List[str]
    question: str
    answer: str
    all_actions: Annotated[List[str], add]


class RAGAssistant:
    def __init__(
        self,
        model_name="Ollama (LLama3.1)",
        embedding_model="nomic-embed-text",
        retrieval_method="similarity",
        chunk_size=500,
        chunk_overlap=50,
        temperature=0.4,
        num_similar_docs=3,
        language="english",
        max_tokens=None
    ):
        self.model_local = get_model(model_name)
        self.embedding_model_name = embedding_model
        self.embedding_model = get_embedding_model(embedding_model)
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

        # If using the custom embedding function for E5 embedding models
        if self.embedding_model_name.startswith("e5-"):
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
        base_kwargs = {"k": self.num_similar_docs}
        if method == "similarity":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=base_kwargs
            )
        elif method == "mmr":
            mmr_kwargs = {**base_kwargs, "fetch_k": 20}
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=mmr_kwargs
            )
        elif method == "similarity_threshold":
            threshold_kwargs = {
                **base_kwargs,
                "score_threshold": 0.8
            }
            search_kwargs = threshold_kwargs
            search_type = "similarity_score_threshold"
            return self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    async def retrieve_context(self, query: str) -> List[Document]:
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None,
            self.retriever.invoke,
            query
        )
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
            | self.model_local.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
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
            msg = (
                "Vector store or retriever not set up. "
                "Call setup_vectorstore() first."
            )
            raise ValueError(msg)

        relevant_docs = await self.retrieve_context(question)
        docs_content = [doc.page_content for doc in relevant_docs]
        context = "\n\n".join(docs_content)

        if prompt_template:
            # Use only the custom prompt template
            custom_prompt = get_prompt_template(prompt_template, self.config)
            rag_chain = custom_prompt | self._get_model_chain()
        else:
            # Use the base template only if no custom template is provided
            base_rag_template = self._get_base_rag_template()
            base_rag_prompt = ChatPromptTemplate.from_template(base_rag_template)
            rag_chain = base_rag_prompt | self._get_model_chain()

        input_dict = {"question": question, "context": context}
        if self.use_history and history:
            input_dict["history"] = _format_history(history)

        response = await rag_chain.ainvoke(input_dict)

        if response is None:
            return self._get_error_message()

        return response
    
    def _get_base_rag_template(self):
        return (
            "Use the following pieces of context to answer the question at "
            "the end. If you don't know the answer, just say that you don't "
            "know, don't try to make up an answer.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def _get_model_chain(self):
        return (
            self.model_local.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            | StrOutputParser()
        )

    def _get_error_message(self):
        return (
            "I apologize, but I couldn't generate a response. "
            "Please try rephrasing your question or providing more context."
        )
