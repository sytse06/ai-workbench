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
    AsyncGenerator
)
import logging
import os
from pathlib import Path

# Third-party imports
import gradio as gr
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)

# Local imports
from .factory import get_model, update_model

logger = logging.getLogger(__name__)

# Functions to support new messaging format Gradiov5
async def format_user_message(
    message: str, 
    history: Optional[List[Dict]] = None, 
    files: Optional[List[gr.File]] = None
) -> Tuple[str, List[Dict]]:
    """
    Format a user message, optionally including file attachments.
    
    Args:
        message: User's text message
        history: Optional chat history
        files: Optional list of file objects
        
    Returns:
        Tuple of (empty string, new history list)
    """
    if history is None:
        history = []
        
    new_history = history.copy()

    if files:
        content = []
        if message and message.strip():
            content.append(message.strip())
            
        for file in files:
            file_path = file.name if hasattr(file, 'name') else str(file)
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different file types
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                content.append({
                    "type": "image",
                    "path": file_path,
                    "alt_text": f"Image: {os.path.basename(file_path)}"
                })
            else:
                content.append({
                    "type": "file",
                    "path": file_path,
                    "alt_text": f"File: {os.path.basename(file_path)}"
                })
                
        new_history.append({"role": "user", "content": content})
    elif message and message.strip():  # Check if message is not empty
        # Simple text message
        new_history.append({"role": "user", "content": message.strip()})
    
    return "", new_history

def format_assistant_message(content: str, metadata: Dict = None) -> Dict:
    """
    Format a message from the assistant.
    
    Args:
        content: Message content
        metadata: Optional metadata dictionary
        
    Returns:
        Formatted message dictionary
    """
    message = {
        "role": "assistant",
        "content": content
    }
    if metadata:
        message["metadata"] = metadata
        
    logger.debug(f"Formatted assistant message: {message}")
    return message

def format_file_content(file_path: str, alt_text: str = None, file_type: str = None) -> dict:
    """Format file content as a properly structured message."""
    if not alt_text:
        alt_text = f"File: {Path(file_path).name}"
        
    file_content = {
        "path": file_path,
        "alt_text": alt_text,
        "type": file_type or Path(file_path).suffix[1:]
    }
    
    return {
        "role": "user",
        "content": [file_content]
    }

def convert_history_to_messages(history: List[Union[BaseMessage, Dict, Tuple[str, str]]]) -> List[Dict]:
    """
    Convert different history formats to Gradio v5 messages format.
    
    Args:
        history: Chat history in various formats (LangChain messages, dicts, or tuples)
        
    Returns:
        List of messages in Gradio v5 format
    """
    messages = []
    
    if not history:
        return messages
        
    for entry in history:
        if isinstance(entry, (HumanMessage, AIMessage, SystemMessage)):
            # Handle LangChain message types
            role = {
                HumanMessage: "user",
                AIMessage: "assistant",
                SystemMessage: "system"
            }.get(type(entry))
            messages.append({
                "role": role,
                "content": entry.content
            })
        elif isinstance(entry, tuple):
            # Handle tuple format (user_msg, assistant_msg)
            user_msg, assistant_msg = entry
            if user_msg:  # Only add non-empty messages
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:  # Only add non-empty messages
                messages.append({"role": "assistant", "content": assistant_msg})
        elif isinstance(entry, dict):
            # Handle dictionary format
            if "role" in entry and "content" in entry:
                messages.append(entry.copy())  # Use copy to avoid modifying original
            elif "speaker" in entry:  # Alternative format
                messages.append({
                    "role": entry["speaker"],
                    "content": entry.get("message", "")
                })
            elif "type" in entry:  # Another alternative format
                messages.append({
                    "role": entry["type"],
                    "content": entry.get("text", "")
                })
    
    return messages

def _format_history(history: List[Union[Dict, Tuple[str, str]]]) -> List[BaseMessage]:
    """
    Convert chat history to LangChain message format.
    
    Args:
        history: List of messages in dictionary or tuple format
        
    Returns:
        List of LangChain message objects
    """
    messages = []
    
    if not history:
        return messages
        
    for entry in history:
        if isinstance(entry, tuple):
            # Handle tuple format (user_msg, assistant_msg)
            user_msg, assistant_msg = entry
            if user_msg:
                messages.append(HumanMessage(content=user_msg))
            if assistant_msg:
                messages.append(AIMessage(content=assistant_msg))
        elif isinstance(entry, dict):
            # Handle dictionary format
            role = entry.get("role", "").lower()
            content = entry.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
                
    return messages

async def process_message(
    message: Dict,
    history: List[Dict],
    model_choice: str,
    prompt_info: Optional[str] = None,
    language_choice: Optional[str] = None,
    history_flag: bool = True,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    files: Optional[List[gr.File]] = None,
    use_context: bool = True
) -> AsyncGenerator[Dict, None]:
    try:
        new_model = await update_model(model_choice, chat_assistant.model_choice)
        if new_model:
            chat_assistant.model = new_model
            chat_assistant.model_choice = model_choice

        result = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=history,
            prompt_info=prompt_info,
            language_choice=language_choice,
            history_flag=history_flag,
            stream=True,
            use_context=use_context
        ):
            result.append(chunk)
            # Format each chunk as a proper message
            formatted_message = format_assistant_message(''.join(result))
            logger.debug(f"Yielding message from process_message: {formatted_message}")  # Log the message
            yield formatted_message
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        error_message = format_assistant_message(f"An error occurred: {str(e)}")
        logger.debug(f"Yielding error message: {error_message}")  # Log the error message
        yield error_message