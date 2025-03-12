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
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

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
    Implementation of message processing for converting between
    Gradio and LangChain message formats.
    """
    
    def format_user_message(
        self,
        message: Union[str, Dict[str, Any]], 
        files: Optional[List[Any]] = None
    ) -> GradioMessage:
        """Format raw user input into a GradioMessage"""
        if isinstance(message, GradioMessage):
            return message
            
        if isinstance(message, dict):
            if "role" in message and message["role"] == "user":
                return GradioMessage(role="user", content=message.get("content", ""))
            elif "text" in message:
                # Handle older style format with text key
                return GradioMessage(role="user", content=message.get("text", ""))
            else:
                return GradioMessage(role="user", content=str(message))
        
        # Default to string conversion
        return GradioMessage(role="user", content=str(message))

    def format_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> GradioMessage:
        """Format assistant response into a GradioMessage"""
        return GradioMessage(role="assistant", content=content)
        
    def format_system_message(
        self, 
        content: str
    ) -> GradioMessage:
        """Format system message into a GradioMessage"""
        return GradioMessage(role="system", content=content)

    async def process_message_content(
        self,
        content: GradioContent
    ) -> str:
        """Process Gradio content into LangChain-compatible string"""
        if isinstance(content, str):
            return content
            
        if isinstance(content, dict) and "path" in content:
            # Handle a single file content
            file_path = content["path"]
            file_type = self._detect_file_type(file_path)
            return f"[{file_type.capitalize()}: {Path(file_path).name}]"
            
        if isinstance(content, list):
            # Handle multimodal content list
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "path" in item:
                    file_path = item["path"]
                    file_type = self._detect_file_type(file_path)
                    text_parts.append(f"[{file_type.capitalize()}: {Path(file_path).name}]")
            
            return " ".join(text_parts)
            
        # Default fallback
        return str(content)

    async def gradio_to_langchain(
        self, 
        message: GradioMessage,
        files: Optional[List[Any]] = None
    ) -> BaseMessage:
        """
        Convert Gradio message format to LangChain format
        
        Args:
            message: GradioMessage object with role and content
            files: Optional list of uploaded files
            
        Returns:
            LangChain BaseMessage subclass
        """
        # Process the content into a LangChain-compatible string
        processed_content = await self.process_message_content(message.content)
        
        # Add additional file content if provided separately
        if files:
            file_descriptions = []
            for file in files:
                if not file or not hasattr(file, "name"):
                    continue
                    
                file_type = self._detect_file_type(file.name)
                file_descriptions.append(f"[{file_type.capitalize()}: {Path(file.name).name}]")
            
            if file_descriptions:
                processed_content += "\n\nAttached files: " + ", ".join(file_descriptions)
        
        # Create appropriate LangChain message based on role
        role = message.role
        if role == "user":
            return HumanMessage(content=processed_content)
        elif role == "assistant":
            return AIMessage(content=processed_content)
        elif role == "system":
            return SystemMessage(content=processed_content)
        else:
            logger.warning(f"Unknown role '{role}', defaulting to HumanMessage")
            return HumanMessage(content=processed_content)

    def langchain_to_gradio(self, message: BaseMessage) -> GradioMessage:
        """
        Convert LangChain message format to Gradio format
        
        Args:
            message: LangChain BaseMessage subclass
            
        Returns:
            GradioMessage object
        """
        content = message.content
        
        # Determine the role based on LangChain message type
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            logger.warning(f"Unknown message type {type(message)}, defaulting to 'user'")
            role = "user"
            
        return GradioMessage(role=role, content=content)
        
    async def convert_history(
        self,
        history: List[Union[GradioMessage, BaseMessage]],
        to_format: Literal["gradio", "langchain"] = "gradio"
    ) -> List[Union[GradioMessage, BaseMessage]]:
        """Convert chat history between different formats"""
        if not history:
            return []
            
        result = []
        
        if to_format == "gradio":
            # Convert to Gradio format
            for msg in history:
                if isinstance(msg, BaseMessage):
                    result.append(self.langchain_to_gradio(msg))
                elif isinstance(msg, dict) and "role" in msg:
                    result.append(GradioMessage(
                        role=msg["role"],
                        content=msg.get("content", "")
                    ))
                elif isinstance(msg, GradioMessage):
                    result.append(msg)
                else:
                    logger.warning(f"Unknown message type in history: {type(msg)}")
        else:
            # Convert to LangChain format
            for msg in history:
                if isinstance(msg, GradioMessage):
                    result.append(await self.gradio_to_langchain(msg))
                elif isinstance(msg, dict) and "role" in msg:
                    gradio_msg = GradioMessage(
                        role=msg["role"],
                        content=msg.get("content", "")
                    )
                    result.append(await self.gradio_to_langchain(gradio_msg))
                elif isinstance(msg, BaseMessage):
                    result.append(msg)
                else:
                    logger.warning(f"Unknown message type in history: {type(msg)}")
                    
        return result
        
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
        Process messages ensuring proper format for both Gradio and LangChain
        
        This is a higher-level method that would typically be implemented by
        specific assistant implementations. Here we provide a basic implementation
        that returns an error message.
        """
        # This method would typically be overridden by specific assistant implementations
        # Here we just yield a placeholder error message
        yield GradioMessage(
            role="assistant",
            content="This is a base implementation. Please use a specific assistant class."
        )

    async def get_message_text(
        self, 
        message: Union[str, Dict, GradioMessage, BaseMessage]
    ) -> str:
        """
        Extract plain text from various message formats
        
        Args:
            message: Message in various formats
            
        Returns:
            Plain text content of the message
        """
        if isinstance(message, str):
            return message
            
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content
                return await self.process_message_content(content)
            return str(content)
            
        if isinstance(message, GradioMessage):
            return await self.process_message_content(message.content)
            
        if hasattr(message, "content"):
            # BaseMessage or similar with content attribute
            return str(message.content)
            
        # Fallback for unknown types
        return str(message)

    def _detect_file_type(self, file_path: str) -> str:
        """Detect if a file is an image or text based on its extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if Path(file_path).suffix.lower() in image_extensions:
            return "image"
        return "text"