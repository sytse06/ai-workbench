# ai_model_core/model_helpers/rag_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging
import asyncio
from pathlib import Path

# Local imports
from .RAG_assistant import RAGAssistant
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.message_types import GradioMessage
from ..shared_utils.factory import update_model as factory_update_model

logger = logging.getLogger(__name__)

class RAGAssistantUI(BaseAssistantUI):
    """
    UI class for RAG Assistant that implements Gradio v5 message structure
    and integrates with the base UI components.
    """
    def __init__(self, rag_assistant: RAGAssistant):
        """
        Initialize the RAG Assistant UI with a reference to the core RAG assistant.
        
        Args:
            rag_assistant: RAGAssistant instance to wrap
        """
        super().__init__(rag_assistant)
        # Store a direct reference to the RAG assistant for easier access
        self.rag_assistant = rag_assistant

    async def process_gradio_message(
        self,
        message: Union[str, Dict],
        history: List[Dict],
        model_choice: str,
        temperature: float,
        max_tokens: int, 
        **kwargs
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Process messages using Gradio v5 format and generate responses."""
        try:
            # Update temperature and max_tokens in the RAGAssistant instance
            self.rag_assistant.temperature = temperature
            self.rag_assistant.max_tokens = max_tokens
            
            # Use message processor from BaseAssistantUI
            message_text = await self.message_processor.get_message_text(message)
            
            # Update model if needed
            if model_choice and model_choice != self.rag_assistant.model_choice:
                await self.rag_assistant.update_model(model_choice)
            
            # Update temperature and max_tokens if provided
            if temperature is not None:
                self.rag_assistant.temperature = temperature
            
            if max_tokens is not None:
                self.rag_assistant.max_tokens = max_tokens
                
            # Use message processor from BaseAssistantUI
            message_text = await self.message_processor.get_message_text(message)
            
            # Convert history if history_flag is True
            langchain_history = []
            if history and history_flag:
                for h in history:
                    if isinstance(h, str):
                        h = {"role": "user", "content": h}
                    if isinstance(h, dict) and "role" in h:
                        msg = GradioMessage(role=h["role"], 
                        content=h.get("content", ""))
                        langchain_history.append(await 
                        self.message_processor.gradio_to_langchain(msg))
                        
            # Log what we're calling query with
            logger.debug(f"Calling RAGAssistant.query with message: {message_text[:30]}...")
            logger.debug(f"History length: {len(langchain_history)}")
            
            # Forward to the RAG assistant's query method
            async for response in self.rag_assistant.query(
                message=message_text,
                history=langchain_history,
                prompt_template=prompt_info
            ):
                if isinstance(response, dict):
                    yield response
                else:
                    yield {"role": "assistant", "content": str(response)}
                    
        except Exception as e:
            logger.error(f"Error in process_gradio_message: {str(e)}")
            yield {
                "role": "assistant",
                "content": f"An error occurred: {str(e)}"
            }
            
    # Enable RAG operations to the underlying RAG assistant
    async def process_uploaded_files(self, files: List[gr.File]) -> None:
        """Process uploaded files and update the vector store."""
        try:
            file_paths = []
            for file in files:
                if hasattr(file, "name"):
                    file_path = Path(file.name)
                    if file_path.exists():
                        file_paths.append(str(file_path))

            if file_paths:
                # Use the RAG assistant's method
                result = await asyncio.to_thread(
                    self.rag_assistant.process_content,
                    url_input="",
                    file_input=file_paths
                )
                logger.info(f"File processing result: {result}")
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            raise
            
    # Property methods to access RAG assistant functionality
    @property
    def is_vectorstore_ready(self):
        """Check if vectorstore is initialized and ready."""
        return self.rag_assistant.is_vectorstore_ready
        
    def reset_vectorstore(self) -> str:
        """Reset the vectorstore and retriever."""
        return self.rag_assistant.reset_vectorstore()
        
    # Enable configuration methods for the RAG assistant
    def set_temperature(self, temperature: float):
        """Set temperature for generation."""
        self.rag_assistant.temperature = temperature
        
    def set_max_tokens(self, max_tokens: int):
        """Set max tokens for generation."""
        self.rag_assistant.max_tokens = max_tokens