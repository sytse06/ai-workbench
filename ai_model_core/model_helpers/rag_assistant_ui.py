# ai_model_core/model_helpers/rag_assistant_ui.py
from typing import Optional, List, Dict, Union, AsyncGenerator
import gradio as gr
import logging
import asyncio
from pathlib import Path

# Local imports
from .base_assistant_ui import BaseAssistantUI
from ..shared_utils.message_processing import MessageProcessor
from ..shared_utils.message_types import GradioMessage

logger = logging.getLogger(__name__)

class RAGAssistantUI(BaseAssistantUI):
    """
    UI class for RAG Assistant using composition pattern with BaseAssistantUI.
    """
    def __init__(self, assistant_obj=None, *args, **kwargs):
        # Initialize message processor
        self.message_processor = MessageProcessor()
        
        # Store the assistant object without type checking
        self._assistant = assistant_obj
        
        # Create the UI helper
        if self._assistant is not None:
            self.ui = BaseAssistantUI(self._assistant)
        else:
            # No UI helper if no assistant
            self.ui = None
            logger.warning("RAGAssistantUI initialized without an assistant")

    async def process_gradio_chat(
        self,
        message: Union[str, Dict],
        history: List[Dict],
        model_choice: str,
        temperature: float,
        max_tokens: int,
        files: Optional[List[gr.File]] = None,
        use_context: bool = True,
        history_flag: bool = True,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, str], None]:
        """
        Process a chat message from Gradio UI for RAG, handles model updates,
        message formatting, and calls the query method.
        """
        try:
            # Update assistant model if needed
            if hasattr(self._assistant, 'update_model'):
                await self._assistant.update_model(model_choice)
            
            # Set parameters if methods exist
            if hasattr(self._assistant, 'set_temperature'):
                self._assistant.set_temperature(temperature)
            elif hasattr(self._assistant, 'temperature'):
                self._assistant.temperature = temperature
                
            if hasattr(self._assistant, 'set_max_tokens'):
                self._assistant.set_max_tokens(max_tokens)
            elif hasattr(self._assistant, 'max_tokens'):
                self._assistant.max_tokens = max_tokens

            # Convert to GradioMessage
            if isinstance(message, str):
                gradio_message = GradioMessage(role="user", content=message)
            elif isinstance(message, dict):
                gradio_message = GradioMessage(role="user", content=message.get("content", ""))
            else:
                gradio_message = GradioMessage(role="user", content=str(message))
                
            # Convert to LangChain format
            langchain_message = await self.message_processor.gradio_to_langchain(gradio_message, files)
            
            # Format history
            langchain_history = []
            if history and history_flag:
                for h in history:
                    if isinstance(h, dict) and "role" in h:
                        h_message = GradioMessage(role=h["role"], content=h.get("content", ""))
                        if h["role"] == "user":
                            langchain_history.append(
                                await self.message_processor.gradio_to_langchain(h_message)
                            )
                        else:
                            langchain_history.append(
                                self.message_processor.format_assistant_message(h_message.content)
                            )

            # Call the query method on the assistant
            if hasattr(self._assistant, 'query'):
                try:
                    query_params = {
                        "message": langchain_message,
                        "history": langchain_history,
                        "prompt_template": prompt_info,
                        "stream": True,
                        "use_context": use_context
                    }
                    
                    # Only include parameters the query method accepts
                    # Check if the query method accepts history_flag
                    if hasattr(self._assistant.query, '__code__') and 'history_flag' in self._assistant.query.__code__.co_varnames:
                        query_params["history_flag"] = history_flag
                    
                    # Check if the query method accepts language_choice
                    if hasattr(self._assistant.query, '__code__') and 'language_choice' in self._assistant.query.__code__.co_varnames:
                        query_params["language_choice"] = language_choice
                    
                    async for response in self._assistant.query(**query_params):
                        # Use UI's formatter if available, otherwise format directly
                        if self.ui:
                            yield self.ui.format_gradio_response(response)
                        else:
                            yield self.format_gradio_response(response)
                except TypeError as e:
                    # If we get a TypeError about unexpected keyword arguments,
                    # try a more basic call without the optional parameters
                    logger.warning(f"Error with full parameters, trying simplified call: {str(e)}")
                    async for response in self._assistant.query(
                        message=langchain_message,
                        history=langchain_history,
                        prompt_template=prompt_info,
                        stream=True,
                        use_context=use_context
                    ):
                        if self.ui:
                            yield self.ui.format_gradio_response(response)
                        else:
                            yield self.format_gradio_response(response)
            else:
                yield {"role": "assistant", "content": "The assistant does not have a query method."}

        except Exception as e:
            logger.error(f"Error in process_gradio_chat: {str(e)}")
            yield {"role": "assistant", "content": f"An error occurred: {str(e)}"}

    # Process_gradio_message method functions as alias for process gradio_chat            
    async def process_gradio_message(self, *args, **kwargs):
        """Delegate to process_gradio_chat for compatibility with existing code"""
        async for response in self.process_gradio_chat(*args, **kwargs):
            yield response
                        
    def format_gradio_response(self, response):
        """Format response for Gradio when UI is not available"""
        if isinstance(response, str):
            return {"role": "assistant", "content": response}
        elif isinstance(response, dict) and "content" in response:
            role = response.get("role", "assistant")
            return {"role": role, "content": response["content"]}
        else:
            return {"role": "assistant", "content": str(response)}
    
    # Enable RAG operations to the underlying RAG assistant
    async def process_uploaded_files(self, files: List[gr.File]) -> str:
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
                if hasattr(self._assistant, 'process_content'):
                    result = await asyncio.to_thread(
                        self._assistant.process_content,
                        url_input="",
                        file_input=file_paths
                    )
                    logger.info(f"File processing result: {result}")
                    return result
                else:
                    message = "Assistant does not support file processing"
                    logger.warning(message)
                    return message
            else:
                return "No valid files to process"
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            return f"Error processing files: {str(e)}"
                    
    # Property methods to access RAG assistant functionality
    @property
    def is_vectorstore_ready(self):
        """Check if vectorstore is initialized and ready."""
        if hasattr(self._assistant, 'is_vectorstore_ready'):
            return self._assistant.is_vectorstore_ready
        return False
        
    def reset_vectorstore(self) -> str:
        """Reset the vectorstore and retriever."""
        if hasattr(self._assistant, 'reset_vectorstore'):
            return self._assistant.reset_vectorstore()
        return "Vectorstore reset not supported"
        
    def set_temperature(self, temperature: float):
        """Set temperature for generation."""
        if hasattr(self._assistant, 'temperature'):
            self._assistant.temperature = temperature
        elif hasattr(self._assistant, 'set_temperature'):
            self._assistant.set_temperature(temperature)
        else:
            logger.warning("Assistant doesn't support setting temperature")
        
    def set_max_tokens(self, max_tokens: int):
        """Set max tokens for generation."""
        if hasattr(self._assistant, 'max_tokens'):
            self._assistant.max_tokens = max_tokens
        elif hasattr(self._assistant, 'set_max_tokens'):
            self._assistant.set_max_tokens(max_tokens)
        else:
            logger.warning("Assistant doesn't support setting max_tokens")

    def set_num_similar_docs(self, num_similar_docs: int):
        """Set number of similar documents to retrieve."""
        if hasattr(self._assistant, 'num_similar_docs'):
            self._assistant.num_similar_docs = num_similar_docs
        else:
            logger.warning("Assistant doesn't support setting num_similar_docs")
            
    def set_retrieval_method(self, method: str):
        """Set retrieval method for the RAG system."""
        if hasattr(self._assistant, 'retrieval_method'):
            self._assistant.retrieval_method = method
            # If the assistant has a method to update the retriever based on method
            if hasattr(self._assistant, 'select_retriever') and hasattr(self._assistant, 'vectorstore'):
                try:
                    self._assistant.retriever = self._assistant.select_retriever(method)
                    logger.info(f"Retriever updated to {method}")
                except Exception as e:
                    logger.error(f"Error updating retriever: {str(e)}")
        else:
            logger.warning("Assistant doesn't support setting retrieval_method")