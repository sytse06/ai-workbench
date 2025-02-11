# Standard library imports
from typing import (
    TypedDict, 
    List, 
    Annotated, 
    Union, 
    Optional, 
    Dict, 
    AsyncGenerator
)
from operator import add
import asyncio
import logging
from pathlib import Path


# Third-party imports
from langgraph.graph import StateGraph, END
from langchain.schema import (
    Document,
    Document, 
    BaseMessage, 
    HumanMessage, 
    AIMessage
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ..shared_utils.factory import (
    get_model,
    get_embedding_model,
    update_model
)
from ..shared_utils.prompt_utils import get_prompt_template
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.utils import EnhancedContentLoader
from ..config.settings import load_config

message_processor = MessageProcessor()


class State(TypedDict):
    input: str
    context: List[str]
    question: str
    answer: str
    all_actions: Annotated[List[str], add]


class RAGAssistant:
    """
    RAG Assistant with persistent vectorstore and integrated message processing.
    Implements singleton pattern to maintain vectorstore persistence across instances.
    """
    _instance = None
    _vectorstore = None  # Class-level vectorstore for persistence
    
    def __new__(cls, *args, **kwargs):
        """Ensure single instance to maintain vectorstore persistence."""
        if cls._instance is None:
            cls._instance = super(RAGAssistant, cls).__new__(cls)
            cls._instance._initialized = False
            cls._graph = StateGraph(State)
        return cls._instance
    def __init__(
        self,
        model_name="Ollama (llama3.2)",
        embedding_model="nomic-embed-text",
        retrieval_method="similarity",
        chunk_size=500,
        chunk_overlap=50,
        temperature=0.4,
        num_similar_docs=3,
        language="english",
        max_tokens: Optional[int] = None
    ):
        """Initialize the RAG Assistant if not already initialized."""
        if self._initialized:
            # Update parameters that can change between instances
            self.model_local = get_model(model_name)
            self.model_choice = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.num_similar_docs = num_similar_docs
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            return

        # First time initialization
        self.model_local = get_model(model_name)
        self.model_choice = model_name
        self.embedding_model_name = embedding_model
        self.embedding_model = get_embedding_model(embedding_model)
        self.temperature = temperature
        self.num_similar_docs = num_similar_docs
        self.language = language
        self.max_tokens = max_tokens
        self.retrieval_method = retrieval_method
        
        # Store chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize message processor
        self.message_processor = MessageProcessor()
        
        # Initialize retriever as None
        self.retriever = None
        
        # Initialize config and state
        self.config = load_config()
        self.use_history = True
        self.prompt_template = None

        # Set up the graph workflow
        self.setup_graph()
        
        self._initialized = True

    def setup_graph(self):
        """Set up the processing graph for the assistant."""
        try:
            self.graph.add_node("retrieve_context", self.retrieve_context)
            self.graph.add_node("generate_answer", self.generate_answer)

            self.graph.add_edge("retrieve_context", "generate_answer")
            self.graph.add_edge("generate_answer", END)

            self.graph_runnable = self.graph.compile()
            logger.info("Graph workflow setup completed")
                
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise
    
    @property
    def graph(self):
        """Access the class-level graph."""
        return self._graph

    def process_content(
        self,
        url_input: str,
        file_input: Union[str, List[str]]
    ) -> str:
        """
        Direct document processing method. Alternative to using the wrapper.
        Useful for different RAG techniques that might need direct control.
        
        Args:
            url_input: URLs to process
            file_input: File paths to process
            
        Returns:
            Status message about the loading process
        """
        try:
            # Use internal content loader if available, or create new one
            content_loader = getattr(self, 'content_loader', None)
            if not content_loader:
                content_loader = EnhancedContentLoader(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )

            # Load and split documents
            docs = content_loader.load_and_split_document(
                file_paths=file_input,
                urls=url_input
            )
            
            # Setup vectorstore with the documents
            self.setup_vectorstore_sync(docs)
            return "Content loaded successfully into memory."
            
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}")
            return f"Error loading content: {str(e)}"


    def setup_vectorstore_sync(self, docs: List[Document]) -> None:
        """
        Synchronous version of vectorstore setup.
        Useful for direct integration and simpler RAG techniques.
        
        Args:
            docs: List of documents to add to vectorstore
        """
        if not docs:
            raise ValueError("No documents were loaded.")

        try:
            # Handle E5 embedding models
            if self.embedding_model_name.startswith("e5-"):
                texts = [doc.page_content for doc in docs]
                embeddings = self.embedding_model.embed_documents(texts)
                text_embeddings = list(zip(texts, embeddings))
                
                # Create or update vectorstore
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_embeddings(
                        text_embeddings=text_embeddings,
                        embedding=self.embedding_model
                    )
                else:
                    self._vectorstore.add_embeddings(
                        text_embeddings=text_embeddings,
                        embedding=self.embedding_model
                    )
            else:
                # Standard embedding models
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(
                        documents=docs,
                        embedding=self.embedding_model
                    )
                else:
                    self._vectorstore.add_documents(documents=docs)

            # Setup retriever
            self.retriever = self.select_retriever(self.retrieval_method)
            logger.info(
                f"Vectorstore setup completed with {len(docs)} documents"
            )
            
        except Exception as e:
            logger.error(f"Error in setup_vectorstore_sync: {str(e)}")
            raise

    async def setup_vectorstore(
        self, 
        docs: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        Asynchronous vectorstore setup used by the wrapper integration.
        Supports chunking parameter updates and async processing.
        
        Args:
            docs: List of documents to add to vectorstore
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
        """
        try:
            if not docs:
                raise ValueError("No documents provided for vector store setup")

            # Update chunking parameters if provided
            if chunk_size is not None:
                self.chunk_size = chunk_size
            if chunk_overlap is not None:
                self.chunk_overlap = chunk_overlap

            logger.info(
                f"Setting up vectorstore with chunk_size={self.chunk_size}, "
                f"chunk_overlap={self.chunk_overlap}"
            )

            # Handle E5 embedding models
            if self.embedding_model_name.startswith("e5-"):
                texts = [doc.page_content for doc in docs]
                embeddings = await asyncio.to_thread(
                    self.embedding_model.embed_documents,
                    texts
                )
                text_embeddings = list(zip(texts, embeddings))
                
                if self._vectorstore is None:
                    self._vectorstore = await asyncio.to_thread(
                        FAISS.from_embeddings,
                        text_embeddings=text_embeddings,
                        embedding=self.embedding_model
                    )
                else:
                    await asyncio.to_thread(
                        self._vectorstore.add_embeddings,
                        text_embeddings=text_embeddings,
                        embedding=self.embedding_model
                    )
            else:
                if self._vectorstore is None:
                    self._vectorstore = await asyncio.to_thread(
                        FAISS.from_documents,
                        documents=docs,
                        embedding=self.embedding_model
                    )
                else:
                    await asyncio.to_thread(
                        self._vectorstore.add_documents,
                        documents=docs
                    )

            # Setup retriever
            self.retriever = self.select_retriever(self.retrieval_method)
            logger.info("Vector store and retriever setup completed successfully")

        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

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

    def generate_answer(self, state: State) -> Dict:
        """
        Generate an answer based on the retrieved context and question.
        Used within the Langgraph workflow.
        
        Args:
            state: Current state containing context and question
            
        Returns:
            Updated state with answer and actions
        """
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

    async def query(
        self,
        message: Union[str, Dict, GradioMessage, BaseMessage],
        history: Optional[List[Union[Dict, GradioMessage, BaseMessage]]] = None,
        stream: bool = True
    ) -> AsyncGenerator[GradioMessage, None]:
        """
        Process a query through the RAG pipeline using Langgraph workflow.
        
        Args:
            message: User message in various formats
            history: Optional chat history
            stream: Whether to stream the response
        """
        try:
            if not self.is_vectorstore_ready:
                yield GradioMessage(
                    role="assistant",
                    content="Vector store not initialized. Please load documents first."
                )
                return

            # Convert message to LangChain format
            if isinstance(message, (str, dict)):
                gradio_message = GradioMessage(role="user", content=message)
                langchain_message = await self.message_processor.gradio_to_langchain(gradio_message)
            elif isinstance(message, GradioMessage):
                langchain_message = await self.message_processor.gradio_to_langchain(message)
            elif isinstance(message, BaseMessage):
                langchain_message = message
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

            # Convert history if provided
            langchain_history = []
            if history and self.use_history:
                langchain_history = await self.message_processor.convert_history(
                    history,
                    to_format="langchain"
                )

            # Get query text
            query = langchain_message.content

            # Initialize graph state
            initial_state = State(
                input=query,
                context=[],
                question=query,
                answer="",
                all_actions=[]
            )

            # Retrieve context and update state
            relevant_docs = await self.retrieve_context(query)
            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
            initial_state["context"] = context_text

            # Run graph workflow
            if stream:
                async for event in self.graph_runnable.astream(initial_state):
                    if "answer" in event:
                        yield GradioMessage(
                            role="assistant",
                            content=event["answer"]
                        )
            else:
                final_state = await self.graph_runnable.ainvoke(initial_state)
                yield GradioMessage(
                    role="assistant",
                    content=final_state["answer"]
                )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            yield GradioMessage(
                role="assistant", 
                content=f"An error occurred: {str(e)}"
            )

    def setup_graph(self):
        """Set up the processing graph for the assistant."""
        try:
            self.graph.add_node("retrieve_context", self.retrieve_context)
            self.graph.add_node("generate_answer", self.generate_answer)

            self.graph.add_edge("retrieve_context", "generate_answer")
            self.graph.add_edge("generate_answer", END)

            self.graph_runnable = self.graph.compile()
            logger.info("Graph workflow setup completed")
                
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise
    
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
