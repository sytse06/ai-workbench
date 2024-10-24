# model_helpers/chat_assistant.py
# Standard library imports
# Standard library imports
import logging
from typing import List, Generator, Any, Optional, Union, Tuple
from pathlib import Path

# Third-party imports
from langchain.schema import HumanMessage, AIMessage, Document
import gradio as gr

# Local imports
from ai_model_core import (
    get_model,
    get_embedding_model,
    get_prompt_template,
    _format_history
)
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader

logger = logging.getLogger(__name__)

class ChatAssistant:
    def __init__(
        self, 
        model_choice: str, 
        temperature: float = 0.7, 
        max_tokens: int = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "input/tmp"
    ):
    def __init__(
        self, 
        model_choice: str, 
        temperature: float = 0.7, 
        max_tokens: int = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "input/tmp"
    ):
        self.model = get_model(model_choice)
        self.model_choice = model_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.documents: List[Document] = []
        
        # Initialize the content loader
        self.content_loader = EnhancedContentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temp_dir=temp_dir
        )

    async def process_chat_context_files(
        self,
        files: List[gr.File]
    ) -> Tuple[str, bool]:
        """
        Process uploaded files for chat context using the internal EnhancedContentLoader.
        
        Args:
            files: List of files uploaded through Gradio interface
            
        Returns:
            Tuple containing:
            - Status message string
            - Success flag (True if processing succeeded)
        """
        if not files:
            return "No files uploaded", False
            
        try:
            # Convert file paths to list
            file_paths = [f.name for f in files]
            
            # Load documents using the existing load_documents method
            success = await self.load_documents(file_paths=file_paths)
            
            if success:
                # Generate status message
                file_names = ", ".join(Path(f.name).name for f in files)
                status_msg = (
                    f"Successfully processed {len(files)} file(s): {file_names}\n"
                    f"Created {len(self.documents)} context chunks for chat"
                )
                
                # Clean up temporary files
                self.content_loader.cleanup()
                
                return status_msg, True
            
            return "Failed to process files", False
            
        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            logger.error(error_msg)
            return error_msg, False

    def get_context_from_docs(
        self,
        message: str,
        use_context: bool = True,
        max_documents: int = 3
    ) -> str:
        """
        Get relevant context from loaded documents based on the message.
        
        Args:
            message: The user's input message
            use_context: Boolean flag to determine if context should be used
            max_documents: Maximum number of documents to include in context
            
        Returns:
            String containing relevant document content if use_context is True,
            empty string otherwise
        """
        if not use_context or not self.documents:
            return ""
            
        return self.get_relevant_context(message, max_documents)
        self.documents: List[Document] = []
        
        # Initialize the content loader
        self.content_loader = EnhancedContentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temp_dir=temp_dir
        )

    async def process_chat_context_files(
        self,
        files: List[gr.File]
    ) -> Tuple[str, bool]:
        """
        Process uploaded files for chat context using the internal EnhancedContentLoader.
        
        Args:
            files: List of files uploaded through Gradio interface
            
        Returns:
            Tuple containing:
            - Status message string
            - Success flag (True if processing succeeded)
        """
        if not files:
            return "No files uploaded", False
            
        try:
            # Convert file paths to list
            file_paths = [f.name for f in files]
            
            # Load documents using the existing load_documents method
            success = await self.load_documents(file_paths=file_paths)
            
            if success:
                # Generate status message
                file_names = ", ".join(Path(f.name).name for f in files)
                status_msg = (
                    f"Successfully processed {len(files)} file(s): {file_names}\n"
                    f"Created {len(self.documents)} context chunks for chat"
                )
                
                # Clean up temporary files
                self.content_loader.cleanup()
                
                return status_msg, True
            
            return "Failed to process files", False
            
        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            logger.error(error_msg)
            return error_msg, False

    def get_context_from_docs(
        self,
        message: str,
        use_context: bool = True,
        max_documents: int = 3
    ) -> str:
        """
        Get relevant context from loaded documents based on the message.
        
        Args:
            message: The user's input message
            use_context: Boolean flag to determine if context should be used
            max_documents: Maximum number of documents to include in context
            
        Returns:
            String containing relevant document content if use_context is True,
            empty string otherwise
        """
        if not use_context or not self.documents:
            return ""
            
        return self.get_relevant_context(message, max_documents)

    def update_model(self, model_choice: str):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice)
            self.model_choice = model_choice

    def _format_history(self, history: List[Tuple[str, str]]) -> List[BaseMessage]:
        formatted_history = []
        for human, ai in history:
            formatted_history.append(HumanMessage(content=human))
            formatted_history.append(AIMessage(content=ai))
        return formatted_history

    async def load_documents(
        self,
        file_paths: Optional[Union[str, List[str], List[Any]]] = None,
        urls: Optional[str] = None
    ) -> bool:
        """
        Load documents from files or URLs and store them in the assistant's memory.
        
        Args:
            file_paths: Single file path or list of file paths to load
            urls: Newline-separated string of URLs to load
            
        Returns:
            bool: True if documents were successfully loaded
        """
        try:
            new_docs = self.content_loader.load_documents(file_paths=file_paths, urls=urls)
            if new_docs:
                self.documents.extend(new_docs)
                logger.info(f"Successfully loaded {len(new_docs)} new documents")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def get_relevant_context(self, message: str, max_documents: int = 3) -> str:
        """
        Retrieve relevant context from loaded documents based on the user's message.
        This is a simple implementation that could be enhanced with embeddings and similarity search.
        
        Args:
            message: The user's input message
            max_documents: Maximum number of documents to include in context
            
        Returns:
            str: Relevant context from documents
        """
        if not self.documents:
            return ""
            
        # Simple keyword matching - could be replaced with more sophisticated retrieval
        relevant_docs = []
        message_words = set(message.lower().split())
        
        for doc in self.documents:
            content = doc.page_content.lower()
            # Count how many query words appear in the document
            relevance_score = sum(1 for word in message_words if word in content)
            if relevance_score > 0:
                relevant_docs.append((relevance_score, doc))
        
        # Sort by relevance and take top documents
        relevant_docs.sort(reverse=True, key=lambda x: x[0])
        selected_docs = relevant_docs[:max_documents]
        
        if not selected_docs:
            return ""
            
        # Format the context
        context_parts = []
        for _, doc in selected_docs:
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ''
            context_parts.append(f"From {source}{page_info}:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)

    async def load_documents(
        self,
        file_paths: Optional[Union[str, List[str], List[Any]]] = None,
        urls: Optional[str] = None
    ) -> bool:
        """
        Load documents from files or URLs and store them in the assistant's memory.
        
        Args:
            file_paths: Single file path or list of file paths to load
            urls: Newline-separated string of URLs to load
            
        Returns:
            bool: True if documents were successfully loaded
        """
        try:
            new_docs = self.content_loader.load_documents(file_paths=file_paths, urls=urls)
            if new_docs:
                self.documents.extend(new_docs)
                logger.info(f"Successfully loaded {len(new_docs)} new documents")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def get_relevant_context(self, message: str, max_documents: int = 3) -> str:
        """
        Retrieve relevant context from loaded documents based on the user's message.
        This is a simple implementation that could be enhanced with embeddings and similarity search.
        
        Args:
            message: The user's input message
            max_documents: Maximum number of documents to include in context
            
        Returns:
            str: Relevant context from documents
        """
        if not self.documents:
            return ""
            
        # Simple keyword matching - could be replaced with more sophisticated retrieval
        relevant_docs = []
        message_words = set(message.lower().split())
        
        for doc in self.documents:
            content = doc.page_content.lower()
            # Count how many query words appear in the document
            relevance_score = sum(1 for word in message_words if word in content)
            if relevance_score > 0:
                relevant_docs.append((relevance_score, doc))
        
        # Sort by relevance and take top documents
        relevant_docs.sort(reverse=True, key=lambda x: x[0])
        selected_docs = relevant_docs[:max_documents]
        
        if not selected_docs:
            return ""
            
        # Format the context
        context_parts = []
        for _, doc in selected_docs:
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ''
            context_parts.append(f"From {source}{page_info}:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)

    async def chat(
        self, 
        message: str, 
        history: List[tuple[str, str]], 
        history: List[tuple[str, str]], 
        history_flag: bool, 
        stream: bool = False,
        use_context: bool = True
        stream: bool = False,
        use_context: bool = True
    ) -> Generator[str, None, None]:
        """
        Enhanced chat function that incorporates document context when relevant.
        
        Args:
            message: User's input message
            history: Chat history as list of (human, ai) message tuples
            history_flag: Whether to include chat history
            stream: Whether to stream the response
            use_context: Whether to include relevant document context
            
        Yields:
            Generated response text
        """
        """
        Enhanced chat function that incorporates document context when relevant.
        
        Args:
            message: User's input message
            history: Chat history as list of (human, ai) message tuples
            history_flag: Whether to include chat history
            stream: Whether to stream the response
            use_context: Whether to include relevant document context
            
        Yields:
            Generated response text
        """
        logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {self.model_choice}")
        
        messages = []
        if history_flag:
            messages.extend(self._format_history(history))
            
        # Add relevant context from documents if available
        if use_context and self.documents:
            context = self.get_context_from_docs(message, use_context)
            if context:
                context_message = (
                    "Here is some relevant information from the uploaded documents:\n\n"
                    f"{context}\n\n"
                    "Please consider this information when responding to the user's message:"
                )
                messages.append(HumanMessage(content=context_message))
                
            
        # Add relevant context from documents if available
        if use_context and self.documents:
            context = self.get_context_from_docs(message, use_context)
            if context:
                context_message = (
                    "Here is some relevant information from the uploaded documents:\n\n"
                    f"{context}\n\n"
                    "Please consider this information when responding to the user's message:"
                )
                messages.append(HumanMessage(content=context_message))
                
        messages.append(HumanMessage(content=message))
        
        # Configure the model with current settings
        # Configure the model with current settings
        self.model = self.model.bind(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            result = await self.model.agenerate([messages])
            yield result.generations[0][0].text

    def clear_documents(self):
        """Clear all loaded documents from memory."""
        self.documents = []
        logger.info("Cleared all documents from memory")

    def get_document_summary(self) -> str:
        """Get a summary of currently loaded documents."""
        if not self.documents:
            return "No documents currently loaded."
            
        summary_parts = []
        sources = {}
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown source')
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
                
        summary_parts.append(f"Total documents loaded: {len(self.documents)}")
        summary_parts.append("\nDocuments by source:")
        for source, count in sources.items():
            summary_parts.append(f"- {source}: {count} document(s)")
            
        return "\n".join(summary_parts)

            yield result.generations[0][0].text

    def clear_documents(self):
        """Clear all loaded documents from memory."""
        self.documents = []
        logger.info("Cleared all documents from memory")

    def get_document_summary(self) -> str:
        """Get a summary of currently loaded documents."""
        if not self.documents:
            return "No documents currently loaded."
            
        summary_parts = []
        sources = {}
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown source')
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
                
        summary_parts.append(f"Total documents loaded: {len(self.documents)}")
        summary_parts.append("\nDocuments by source:")
        for source, count in sources.items():
            summary_parts.append(f"- {source}: {count} document(s)")
            
        return "\n".join(summary_parts)

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens