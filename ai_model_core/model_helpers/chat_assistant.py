# model_helpers/chat_assistant.py
# Standard library imports
import logging
import os
from typing import (
    List,
    Optional,
    Union,
    Dict,
    AsyncGenerator
)

# Third-party imports
from langchain.schema import (
    HumanMessage,
    AIMessage,
    Document,
    BaseMessage,
    SystemMessage
)

# Local imports
from ..config.settings import load_config
from ..shared_utils.factory import get_model, update_model
from ..shared_utils.prompt_utils import (
    get_prompt_template, 
    get_system_prompt
)
from ..shared_utils.message_types import (
    GradioMessage,
    GradioContent
)
from ..shared_utils.message_processing import MessageProcessor

logger = logging.getLogger(__name__)

class ChatAssistant:
    """
    Chat assistant that handles conversation with LLMs and manages context from documents.
    """
    
    def __init__(
        self, 
        model_choice: str, 
        temperature: float = 0.7, 
        max_tokens: int = None
    ):
        """
        Initialize ChatAssistant with model and generation parameters.
        
        Args:
            model_choice: Name of the model to use
            temperature: Temperature for text generation
            max_tokens: Maximum tokens in response
        """
        self.model = get_model(model_choice)
        self.model_choice = model_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.documents: List[Document] = []
        self.config = load_config()
        self.message_processor = MessageProcessor()
        
    async def update_model(self, model_choice: str) -> None:
        """
        Update the model if the model choice has changed.
        
        Args:
            model_choice: New model name
        """
        if self.model_choice != model_choice:
            new_model = await update_model(model_choice, self.model_choice)
            if new_model:
                self.model = new_model 
                self.model_choice = model_choice
                logger.info(f"Model updated to {model_choice}")
        
    async def process_documents(self, docs: List[Document]) -> None:
        """
        Process documents for chat context - called by ContentProcessingComponent.
        This method maintains state by storing documents in the assistant instance.
        
        Args:
            docs: List of Document objects loaded and chunked by ContentProcessingComponent
        """
        try:
            # Store the documents in the assistant instance
            self.documents = docs
            logger.info(f"Processed {len(docs)} documents for chat context")
            
        except Exception as e:
            logger.error(f"Error processing documents for chat: {str(e)}")
            raise

    def get_context_from_docs(
        self,
        message: str,
        use_context: bool = True,
        max_documents: int = 3
    ) -> str:
        """
        Get relevant context from loaded documents.
        
        Args:
            message: User message to find relevant context for
            use_context: Whether to use document context
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted context string
        """
        if not use_context or not self.documents:
            return ""
            
        return self.get_relevant_context(message, max_documents)

    def get_relevant_context(self, message: str, max_documents: int = 3) -> str:
        """
        Retrieve relevant context based on the message.
        
        Args:
            message: User message to find relevant context for
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted context string with most relevant documents
        """
        if not self.documents:
            return ""
            
        relevant_docs = []
        message_words = set(message.lower().split())
        
        for doc in self.documents:
            content = doc.page_content.lower()
            relevance_score = sum(1 for word in message_words if word in content)
            if relevance_score > 0:
                relevant_docs.append((relevance_score, doc))
        
        relevant_docs.sort(reverse=True, key=lambda x: x[0])
        selected_docs = relevant_docs[:max_documents]
        
        if not selected_docs:
            return ""
            
        context_parts = []
        for _, doc in selected_docs:
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ''
            context_parts.append(f"From {source}{page_info}:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)

    async def chat(
        self,
        message: Union[Dict, str, GradioMessage, HumanMessage],
        history: List[Union[Dict, GradioMessage, BaseMessage]] = None,
        history_flag: bool = True,
        stream: bool = False,
        use_context: bool = True,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Chat method handling both Gradio and LangChain message formats.
        
        Args:
            message: Input message (various formats supported)
            history: Chat history in various formats
            history_flag: Whether to include history
            stream: Whether to stream responses
            use_context: Whether to use context
            prompt_info: Optional prompt template
            language_choice: Optional language choice
        
        Yields:
            Generated response text, streaming if requested
        """
        try:
            # Get message text for context matching
            message_text = await self.message_processor.get_message_text(message)
            
            # Convert message to LangChain format
            if not isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                langchain_message = await self.message_processor.gradio_to_langchain(message)
            else:
                langchain_message = message

            # Convert history to LangChain format if needed
            langchain_history = []
            if history_flag and history:
                langchain_history = await self.message_processor.convert_history(
                    history, to_format="langchain"
                )

            # Prepare messages list
            messages = []
            
            # Add system message if using prompts
            if prompt_info and language_choice:
                system_prompt = get_system_prompt(language_choice, self.config)
                messages.append(SystemMessage(content=system_prompt))

            # Add history
            if history_flag and langchain_history:
                messages.extend(langchain_history)

            # Add relevant context if enabled
            if use_context and self.documents:
                context = self.get_context_from_docs(message_text, use_context)
                if context:
                    messages.append(SystemMessage(content=f"Context:\n{context}"))

            # Add current message
            messages.append(langchain_message)

            # Configure model parameters
            self._configure_model(stream)

            # Generate response
            if stream:
                async for chunk in self.model.astream(messages):
                    yield chunk.content
            else:
                result = await self.model.agenerate([messages])
                yield result.generations[0][0].text

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def _configure_model(self, stream: bool = False):
        """
        Configure model parameters based on model type.
        
        Args:
            stream: Whether to enable streaming
        """
        if "ollama" in self.model_choice.lower():
            self.model = self.model.bind(
                stop=None,
                stream=stream
            )
        elif "gemini" in self.model_choice.lower():
            self.model = self.model.bind(
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
        else:
            self.model = self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
    def clear_documents(self):
        """Clear all loaded documents."""
        self.documents = []
        logger.info("Cleared all documents from memory")

    def get_document_summary(self) -> str:
        """
        Get a summary of loaded documents.
        
        Returns:
            Summary string showing document count and sources
        """
        if not self.documents:
            return "No documents currently loaded."
            
        summary_parts = []
        sources = {}
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown source')
            sources[source] = sources.get(source, 0) + 1
                
        summary_parts.append(f"Total documents loaded: {len(self.documents)}")
        summary_parts.append("\nDocuments by source:")
        for source, count in sources.items():
            summary_parts.append(f"- {source}: {count} document(s)")
            
        return "\n".join(summary_parts)

    def set_temperature(self, temperature: float):
        """Set the temperature parameter."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        """Set the max_tokens parameter."""
        self.max_tokens = max_tokens