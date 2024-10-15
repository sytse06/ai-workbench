# model_helpers/RAG_assistant.py
# Standard library imports
import asyncio
from typing import TypedDict, List, Annotated, Union
from operator import add

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModel
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ai_model_core import (
    get_model,
    get_embedding_model,
    get_prompt_template,
    _format_history
)
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader


class State(TypedDict):
    input: str
    context: List[str]
    question: str
    answer: str
    all_actions: Annotated[List[str], add]


class CustomHuggingFaceEmbeddings:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt'
        )
        encoded_input = {
            k: v.to(self.device) for k, v in encoded_input.items()
        }
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        )
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


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
        self.content_loader = EnhancedContentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.retrieval_method = retrieval_method

    def process_content(self, url_input: str, file_input: Union[str, List[str]]):
        try:
            docs = self.content_loader.load_and_split_document(
                file_paths=file_input, urls=url_input
            )
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
            text_embeddings = list(zip(texts, embeddings))
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
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
            threshold_kwargs = {**base_kwargs, "score_threshold": 0.8}
            return self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=threshold_kwargs
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    async def retrieve_context(self, query: str) -> List[Document]:
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
            raise ValueError(
                "Vector store or retriever not set up. "
                "Call setup_vectorstore() first."
            )

        relevant_docs = await self.retrieve_context(question)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        base_rag_template = self._get_base_rag_template()
        base_rag_prompt = ChatPromptTemplate.from_template(base_rag_template)

        if prompt_template:
            custom_prompt = get_prompt_template(prompt_template, self.config)
            rag_chain = self._create_custom_chain(
                base_rag_prompt, custom_prompt
            )
        else:
            rag_chain = self._create_base_chain(base_rag_prompt)

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

    def _create_custom_chain(self, base_rag_prompt, custom_prompt):
        context_and_question = {
            "context": RunnablePassthrough(),
            "question": custom_prompt
        }
        return (
            context_and_question
            | base_rag_prompt
            | self._get_model_chain()
        )

    def _create_base_chain(self, base_rag_prompt):
        return base_rag_prompt | self._get_model_chain()

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