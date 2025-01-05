# model_helpers/chat_assistant.py
# Standard library imports
import logging
from pathlib import Path
import os
from typing import List, Generator, Any, Optional, Union, Tuple

# Third-party imports
from langchain.schema import HumanMessage, AIMessage, Document, BaseMessage
import gradio as gr

# Local imports
from ..shared_utils.factory import (
    get_model,
    get_embedding_model
)
from ..config.settings import load_config
from ..shared_utils.utils import (
    EnhancedContentLoader,
    get_prompt_template,
    _format_history
)

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "AI-Workbench/1.0"

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

    async def update_model(self, model_choice: str):
        """Asynchronously update the model if the choice has changed."""
        if self.model_choice != model_choice:
            self.model = get_model(model_choice)
            self.model_choice = model_choice

    async def process_chat_context_files(
        self,
        files: List[gr.File]
    ) -> Tuple[str, bool]:
        """Process uploaded files for chat context using the internal EnhancedContentLoader."""
        if not files:
            return "No files uploaded", False
            
        try:
            file_paths = [f.name for f in files]
            success = await self.load_documents(file_paths=file_paths)
            
            if success:
                file_names = ", ".join(Path(f.name).name for f in files)
                status_msg = (
                    f"Successfully processed {len(files)} file(s): {file_names}\n"
                    f"Created {len(self.documents)} context chunks for chat"
                )
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
        """Get relevant context from loaded documents based on the message."""
        if not use_context or not self.documents:
            return ""
            
        return self.get_relevant_context(message, max_documents)

    async def load_documents(
        self,
        file_paths: Optional[Union[str, List[str], List[Any]]] = None,
        urls: Optional[str] = None
    ) -> bool:
        """Load documents from files or URLs and store them in the assistant's memory."""
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
        """Retrieve relevant context from loaded documents based on the user's message."""
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
        message: str, 
        history: List[tuple[str, str]], 
        history_flag: bool, 
        stream: bool = False,
        use_context: bool = True
    ) -> Generator[str, None, None]:
        """Enhanced chat function that incorporates document context when relevant."""
        logger.info(f"Chat function called with message: {message}, history_flag: {history_flag}, model_choice: {self.model_choice}")
        
        messages = []
        if history_flag and history:
            try:
                formatted_messages = _format_history(history)
                messages.extend(formatted_messages)
                logger.info(f"Successfully formatted {len(formatted_messages)} messages from history")
            except Exception as e:
                logger.error(f"Error formatting history: {str(e)}")
                logger.error(f"History that caused error: {history}")
                pass
                
        messages.append(HumanMessage(content=message))
        
        # Configure model based on type
        if "ollama" in self.model_choice.lower():
            if stream:
                self.model = self.model.bind(
                    stop=None,
                    stream=True
                )
            else:
                self.model = self.model.bind(
                    stop=None
                )
        elif "gemini" in self.model_choice.lower():
            # Gemini uses generation_config without stream parameter
            self.model = self.model.bind(
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
        else:
            if stream:
                self.model = self.model.bind(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
            else:
                self.model = self.model.bind(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
        
        try:
            if stream and not "gemini" in self.model_choice.lower():
                async for chunk in self.model.astream(messages):
                    yield chunk.content
            else:
                result = await self.model.agenerate([messages])
                yield result.generations[0][0].text
        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            yield f"An error occurred: {str(e)}"
        
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
        """Set the temperature parameter for the model."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        """Set the max_tokens parameter for the model."""
        self.max_tokens = max_tokens
