# ai_model_core/shared_utils/message_processing.py
# Standard library imports
from typing import (
    List,
    Generator,
    Any,
    Optional,
    Union,
    Tuple,
    Dict,
    Generator, 
    AsyncGenerator,
    Literal
)
import logging
from pathlib import Path
import mimetypes

# Third-party imports
import gradio as gr
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)
import asyncio

# Local imports
from .message_types import (
    BaseMessageProcessor,
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)

logger = logging.getLogger(__name__)

# Functions to support new messaging format Gradiov5
class MessageProcessor(BaseMessageProcessor):
    """
    Handles conversion between different message formats (Gradio, LangChain).
    
    This class facilitates bidirectional conversion between the message formats
    used by Gradio and those used by LangChain, enabling seamless integration
    between the UI and LLM components.
    """
    
    def format_user_message(
        self,
        message: Union[str, Dict[str, Any]], 
        files: Optional[List[Any]] = None
    ) -> GradioMessage:
        """
        Format raw user input into a GradioMessage.
        
        Args:
            message: User message content (string or dict)
            files: Optional list of file objects
            
        Returns:
            Formatted GradioMessage
        """
        try:
            content: GradioContent
            
            if isinstance(message, str):
                # Simple text message
                content = message
            elif isinstance(message, dict):
                # Already formatted message
                if "role" in message and "content" in message:
                    return GradioMessage.from_dict(message)
                # Content as dict
                if "text" in message:
                    content = message["text"]
                else:
                    content = str(message)
            else:
                # Unknown format, convert to string
                content = str(message)
            
            # Process files if provided
            if files:
                # Create multimodal content
                if isinstance(content, str) and content:
                    # Text + files
                    mixed_content: List[Union[str, GradioFileContent]] = [content]
                    for file in files:
                        if hasattr(file, 'name'):
                            file_path = file.name
                        else:
                            file_path = str(file)
                        mixed_content.append({"path": file_path})
                    content = mixed_content
                else:
                    # Files only
                    file_contents = []
                    for file in files:
                        if hasattr(file, 'name'):
                            file_path = file.name
                        else:
                            file_path = str(file)
                        file_contents.append({"path": file_path})
                    content = file_contents
            
            return GradioMessage(role="user", content=content)
            
        except Exception as e:
            logger.error(f"Error formatting user message: {str(e)}")
            # Fallback to simple text message
            return GradioMessage(role="user", content=str(message))
    
    def format_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> GradioMessage:
        """
        Format assistant response into a GradioMessage.
        
        Args:
            content: Assistant response text
            metadata: Optional metadata to include
            
        Returns:
            Formatted GradioMessage
        """
        return GradioMessage(role="assistant", content=content)
    
    def format_system_message(
        self, 
        content: str
    ) -> GradioMessage:
        """
        Format system message into a GradioMessage.
        
        Args:
            content: System message text
            
        Returns:
            Formatted GradioMessage
        """
        return GradioMessage(role="system", content=content)
    
    async def process_message_content(
        self,
        content: GradioContent
    ) -> str:
        """
        Process Gradio content into LangChain-compatible string.
        
        Args:
            content: Gradio content (text, file, or mixed)
            
        Returns:
            Processed string content
        """
        try:
            # Handle different content types
            if isinstance(content, str):
                # Simple text
                return content
                
            elif isinstance(content, dict) and "path" in content:
                # Single file
                file_path = content["path"]
                alt_text = content.get("alt_text", f"File: {Path(file_path).name}")
                return f"[{alt_text}] (File: {file_path})"
                
            elif isinstance(content, list):
                # Mixed content
                text_parts = []
                file_parts = []
                
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict) and "path" in item:
                        file_path = item["path"]
                        alt_text = item.get("alt_text", f"File: {Path(file_path).name}")
                        file_parts.append(f"[{alt_text}] (File: {file_path})")
                
                # Combine parts
                result = " ".join(text_parts)
                if file_parts:
                    file_section = "\n".join(file_parts)
                    if result:
                        result += f"\n\n{file_section}"
                    else:
                        result = file_section
                
                return result
                
            else:
                # Unknown format
                return str(content)
                
        except Exception as e:
            logger.error(f"Error processing message content: {str(e)}")
            return str(content)
    
    async def gradio_to_langchain(self, message):
        """Convert Gradio message format to LangChain format."""
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        
        try:
            # Already a LangChain message
            if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
                return message
                
            # Convert simple string to HumanMessage
            if isinstance(message, str):
                return HumanMessage(content=message)
                
            # Convert GradioMessage to LangChain
            if isinstance(message, GradioMessage):
                content = await self.get_message_text(message)
                
                if message.role == GradioRole.USER:
                    return HumanMessage(content=content)
                elif message.role == GradioRole.ASSISTANT:
                    return AIMessage(content=content)
                elif message.role == GradioRole.SYSTEM:
                    return SystemMessage(content=content)
                    
            # Convert dictionary format
            if isinstance(message, dict):
                content = await self.get_message_text(message)
                role = message.get("role", "user").lower()
                
                if role == "user":
                    return HumanMessage(content=content)
                elif role == "assistant":
                    return AIMessage(content=content)
                elif role == "system":
                    return SystemMessage(content=content)
            
            # Default fallback
            return HumanMessage(content=str(message))
            
        except Exception as e:
            logger.warning(f"Error converting message: {e}")
            return HumanMessage(content=str(message))
        
    def langchain_to_gradio(
        self,
        message: BaseMessage
    ) -> GradioMessage:
        """
        Convert LangChain message to Gradio format.
        
        Args:
            message: LangChain message
            
        Returns:
            GradioMessage
        """
        try:
            # Extract content
            content = message.content
            
            # Map LangChain message type to Gradio role
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                logger.warning(f"Unknown LangChain message type: {type(message)}")
                role = "user"  # Default to user role
            
            # Return Gradio message
            return GradioMessage(role=role, content=content)
            
        except Exception as e:
            logger.error(f"Error converting LangChain to Gradio: {str(e)}")
            # Fallback
            return GradioMessage(role="user", content=str(message))
    
    async def convert_history(
        self,
        history: List[Union[GradioMessage, BaseMessage, Dict]],
        to_format: Literal["gradio", "langchain"] = "gradio"
    ) -> List[Union[GradioMessage, BaseMessage]]:
        """
        Convert chat history between different formats.
        
        Args:
            history: List of messages
            to_format: Target format ("gradio" or "langchain")
            
        Returns:
            Converted message history
        """
        converted = []
        
        if not history:
            return []
            
        try:
            for message in history:
                if to_format.lower() == "langchain":
                    # Convert to LangChain format
                    if isinstance(message, BaseMessage):
                        # Already in LangChain format
                        converted.append(message)
                    elif isinstance(message, GradioMessage):
                        # Convert GradioMessage to LangChain
                        converted.append(await self.gradio_to_langchain(message))
                    elif isinstance(message, dict):
                        # Convert dict to GradioMessage, then to LangChain
                        gradio_msg = GradioMessage.from_dict(message)
                        converted.append(await self.gradio_to_langchain(gradio_msg))
                    else:
                        # Unknown format
                        logger.warning(f"Unknown message format: {type(message)}")
                        converted.append(HumanMessage(content=str(message)))
                        
                elif to_format.lower() == "gradio":
                    # Convert to Gradio format
                    if isinstance(message, BaseMessage):
                        # Convert from LangChain format
                        converted.append(self.langchain_to_gradio(message))
                    elif isinstance(message, GradioMessage):
                        # Already in GradioMessage format
                        converted.append(message)
                    elif isinstance(message, dict) and "role" in message:
                        # Dict format - convert to GradioMessage
                        converted.append(GradioMessage.from_dict(message))
                    else:
                        # Unknown format
                        logger.warning(f"Unknown message format: {type(message)}")
                        converted.append(GradioMessage(role="user", content=str(message)))
            
            return converted
            
        except Exception as e:
            logger.error(f"Error converting history: {str(e)}")
            return []
    
    async def process_message(
        self,
        message: GradioMessage,
        history: List[GradioMessage],
        model_choice: str,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None,
        history_flag: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        files: Optional[List[Any]] = None,
        use_context: bool = True
    ) -> AsyncGenerator[GradioMessage, None]:
        """
        Process messages ensuring proper format for both Gradio and LangChain.
        
        This is a placeholder implementation - should be implemented by a subclass
        that has access to an LLM or assistant implementation.
        
        Args:
            message: User message
            history: Chat history
            model_choice: Model to use
            prompt_info: Optional prompt template
            language_choice: Language for response
            history_flag: Whether to include history
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            files: Optional files to include
            use_context: Whether to use context
            
        Yields:
            Assistant response messages
        """
        # This is a placeholder - this should be implemented in assistant-specific classes
        logger.warning("process_message called on base MessageProcessor")
        yield GradioMessage(
            role="assistant",
            content="This method should be implemented by an assistant-specific processor."
        )
    
    async def get_message_text(self, message):
        """Extract text content from various message formats."""
        try:
            # Handle basic string
            if isinstance(message, str):
                return message
                
            # Handle dict format (common in API calls)
            if isinstance(message, dict):
                # Handle Gradio v5 format
                if "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Extract text from multimodal content
                        text_parts = []
                        for item in content:
                            if isinstance(item, str):
                                text_parts.append(item)
                        return " ".join(text_parts)
                    return str(content)
                # Handle older format
                elif "text" in message:
                    return message["text"]
            
            # Default fallback
            return str(message)
            
        except Exception as e:
            logger.warning(f"Error extracting message text: {e}")
            return str(message)
    
    def get_langchain_message_text(self, message):
        """Extract text content from LangChain message types."""
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
            return message.content
        # For any other type, use the standard method
        return self.get_message_text(message)
        
    def _extract_text_from_content_list(self, content_list):
        """Extract text parts from a multimodal content list."""
        text_parts = []
        for item in content_list:
            if isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    
    async def _process_files(self, files: Optional[List[Any]] = None) -> List[Dict[str, str]]:
        """
        Process file attachments for inclusion in messages.
        
        Args:
            files: List of file objects or file paths
            
        Returns:
            List of processed file metadata
        """
        if not files:
            return []
            
        processed_files = []
        
        for file in files:
            try:
                # Handle different file types
                if hasattr(file, 'name'):
                    file_path = file.name
                else:
                    file_path = str(file)
                    
                # Add as a file reference
                processed_files.append({
                    "type": "file_path", 
                    "path": file_path,
                    "mime_type": self._guess_mime_type(file_path)
                })
                    
            except Exception as e:
                logger.error(f"Error processing file {file}: {str(e)}")
                
        return processed_files
    
    def _guess_mime_type(self, file_path: str) -> str:
        """Guess the MIME type of a file based on its extension."""
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Default to application/octet-stream if unknown
        if mime_type is None:
            mime_type = "application/octet-stream"
            
        return mime_type