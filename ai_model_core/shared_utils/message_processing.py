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

# Third-party imports
import gradio as gr
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)

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
async def format_user_message(
    message: str, 
    history: Optional[List[Dict]] = None, 
    files: Optional[List[gr.File]] = None
)  -> Dict[str, Any]:
    """
    Format a user message, optionally including file attachments.
    
    Args:
        message: User's text message
        history: Optional chat history
        files: Optional list of file objects
        
    Returns:
        Tuple of (empty string, new history list)
    """
    # Initialize content list for multimodal messages
    content: Union[str, List[Union[str, Dict]]] = []
    
    # Handle text content
    if isinstance(message, str):
        if message.strip():
            content = message.strip()
    elif isinstance(message, dict):
        if "text" in message and message["text"]:
            content = message["text"].strip()
            
    # Handle file attachments
    if files:
        # Convert content to list if it's a string and not empty
        if isinstance(content, str) and content:
            content = [content]
        elif isinstance(content, str):
            content = []
            
        # Add file content
        for file in files:
            file_path = file.name if hasattr(file, 'name') else str(file)
            file_ext = Path(file_path).suffix.lower()
            
            file_content = {
                "path": file_path,
                "type": "image" if file_ext in ['.jpg', '.jpeg', '.png', '.gif'] else "file",
                "alt_text": f"{'Image' if file_ext in ['.jpg', '.jpeg', '.png', '.gif'] else 'File'}: {os.path.basename(file_path)}"
            }
            
            if isinstance(content, list):
                content.append(file_content)
            else:
                content = [content, file_content]

    return {
        "role": "user",
        "content": content if isinstance(content, list) else (content or "")
    }

def format_assistant_message(
    content: str, 
    metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
    """
    Format an assistant message into Gradio v5 format.
    
    Args:
        content: Message content
        metadata: Optional metadata
        
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

def format_system_message(content: str) -> Dict[str, str]:
    """
    Format a system message into Gradio v5 format.
    
    Args:
        content: System message content
        
    Returns:
        Formatted message dictionary
    """
    return {
        "role": "system",
        "content": content
    }

def convert_gradio_to_langchain(
    message: Dict[str, Any]
) -> BaseMessage:
    """
    Convert a Gradio message to LangChain format.
    
    Args:
        message: Message in Gradio v5 format
        
    Returns:
        LangChain message object
    """
    role = message["role"]
    content = message["content"]

    # Handle multimodal content
    if isinstance(content, list):
        processed_content = []
        for item in content:
            if isinstance(item, str):
                processed_content.append(item)
            elif isinstance(item, dict):
                if "path" in item:
                    processed_content.append(f"[{item.get('type', 'File')}: {item['path']}]")
                elif "text" in item:
                    processed_content.append(item["text"])
        content = " ".join(processed_content)

    # Create appropriate message type
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"Unknown role type: {role}")

def convert_langchain_to_gradio(
    message: BaseMessage
) -> Dict[str, str]:
    """
    Convert a LangChain message to Gradio format.
    
    Args:
        message: LangChain message object
        
    Returns:
        Message in Gradio v5 format
    """
    role_mapping = {
        HumanMessage: "user",
        AIMessage: "assistant",
        SystemMessage: "system"
    }
    
    role = role_mapping.get(type(message))
    if not role:
        raise ValueError(f"Unsupported message type: {type(message)}")
        
    return {
        "role": role,
        "content": message.content
    }

def convert_history(
    history: List[Union[BaseMessage, Dict, tuple]],
    to_format: str = "gradio"
) -> List[Union[Dict, BaseMessage]]:
    """
    Unified function to convert chat history between formats.
    Replaces convert_history_to_messages and _format_history.
    
    Args:
        history: Chat history in any supported format
        to_format: Target format ("gradio" or "langchain")
        
    Returns:
        Converted message list in specified format
    """
    if not history:
        return []

    converted_history = []
    
    for entry in history:
        if to_format == "gradio":
            if isinstance(entry, BaseMessage):
                # LangChain message to Gradio format
                role = {
                    HumanMessage: "user",
                    AIMessage: "assistant",
                    SystemMessage: "system"
                }.get(type(entry))
                converted_history.append({
                    "role": role,
                    "content": entry.content
                })
            elif isinstance(entry, tuple):
                # Tuple format to Gradio format
                user_msg, assistant_msg = entry
                if user_msg:
                    converted_history.append({
                        "role": "user",
                        "content": user_msg
                    })
                if assistant_msg:
                    converted_history.append({
                        "role": "assistant",
                        "content": assistant_msg
                    })
            elif isinstance(entry, dict) and "role" in entry and "content" in entry:
                # Already in Gradio format
                converted_history.append(entry.copy())
                
        elif to_format == "langchain":
            if isinstance(entry, dict):
                # Gradio format to LangChain
                role = entry["role"]
                content = entry["content"]
                
                if isinstance(content, list):
                    # Handle multimodal content
                    content = " ".join(
                        item if isinstance(item, str)
                        else f"[{item.get('type', 'File')}: {item['path']}]"
                        for item in content
                    )
                    
                if role == "user":
                    converted_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    converted_history.append(AIMessage(content=content))
                elif role == "system":
                    converted_history.append(SystemMessage(content=content))
                    
            elif isinstance(entry, BaseMessage):
                # Already in LangChain format
                converted_history.append(entry)
            elif isinstance(entry, tuple):
                # Tuple format to LangChain
                user_msg, assistant_msg = entry
                if user_msg:
                    converted_history.append(HumanMessage(content=user_msg))
                if assistant_msg:
                    converted_history.append(AIMessage(content=assistant_msg))
                    
    return converted_history

async def process_message(
    message: Dict[str, Any],
    history: List[Dict],
    model_choice: str,
    prompt_info: Optional[str] = None,
    language_choice: Optional[str] = None,
    history_flag: bool = True,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    files: Optional[List[gr.File]] = None,
    use_context: bool = True
) -> AsyncGenerator[Union[str, Dict[str, str]], None]:
    """Process messages ensuring proper format for both Gradio and LangChain."""
    try:
        # Convert Gradio message to LangChain format
        content = message.get("content", "")
        if isinstance(content, list):
            # Handle multimodal content
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "path" in item:
                    text_parts.append(f"[File: {item['path']}]")
            content = " ".join(text_parts)
        langchain_message = HumanMessage(content=content)

        # Convert history to LangChain format
        langchain_history = []
        if history_flag and history:
            for h in history:
                if not isinstance(h, dict):
                    continue
                    
                h_content = h.get("content", "")
                if isinstance(h_content, list):
                    # Handle multimodal content in history
                    h_content = " ".join(
                        item if isinstance(item, str)
                        else f"[File: {item['path']}]"
                        for item in h_content
                    )
                
                if h["role"] == "user":
                    langchain_history.append(HumanMessage(content=h_content))
                elif h["role"] == "assistant":
                    langchain_history.append(AIMessage(content=h_content))

        # Get response from chat assistant
        async for chunk in chat_assistant.chat(
            message=langchain_message,
            history=langchain_history,
            prompt_info=prompt_info,
            language_choice=language_choice,
            history_flag=history_flag,
            stream=True,
            use_context=use_context
        ):
            # Return properly formatted messages
            if isinstance(chunk, str):
                yield {"role": "assistant", "content": chunk}
            elif isinstance(chunk, dict):
                yield chunk
            else:
                yield {"role": "assistant", "content": str(chunk)}

    except Exception as e:
        logger.error(f"Process message error: {str(e)}")
        yield {"role": "assistant", "content": f"An error occurred: {str(e)}"}