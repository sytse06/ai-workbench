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
class MessageProcessor(BaseMessageProcessor):
    """Implementation of message processing logic"""
    
    async def process_message_content(
        self,
        content: GradioContent
    ) -> str:
        """Convert Gradio's flexible content types to LangChain string format"""
        if isinstance(content, str):
            return content.strip()
            
        if isinstance(content, dict) and "path" in content:
            # Single file content
            try:
                async with aiofiles.open(content["path"], 'rb') as f:
                    await f.read(1)  # Verify file is readable
                return f"[File: {content['path']}]"
            except Exception as e:
                logger.error(f"Error accessing file {content['path']}: {str(e)}")
                return ""
                
        if isinstance(content, list):
            # Mixed content (text + files)
            processed_parts = []
            for item in content:
                if isinstance(item, str):
                    processed_parts.append(item.strip())
                elif isinstance(item, dict) and "path" in item:
                    try:
                        async with aiofiles.open(item["path"], 'rb') as f:
                            await f.read(1)
                        processed_parts.append(f"[File: {item['path']}]")
                    except Exception as e:
                        logger.error(f"Error accessing file {item['path']}: {str(e)}")
                        continue
            return " ".join(processed_parts)
            
        return ""  # Return empty string for unsupported content types

    async def gradio_to_langchain(
        self,
        message: GradioMessage
    ) -> BaseMessage:
        """Convert Gradio message to LangChain format"""
        content = await self.process_message_content(message.content)
        
        if message.role == "user":
            return HumanMessage(content=content)
        elif message.role == "assistant":
            return AIMessage(content=content)
        elif message.role == "system":
            return SystemMessage(content=content)
        else:
            raise ValueError(f"Unsupported role: {message.role}")

    def langchain_to_gradio(
        self,
        message: BaseMessage
    ) -> GradioMessage:
        """Convert LangChain message to Gradio format"""
        role: GradioRole
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
            
        return GradioMessage(role=role, content=message.content)
        
    async def convert_history(
        self,
        history: List[Union[GradioMessage, BaseMessage]],
        to_format: Literal["gradio", "langchain"] = "gradio"
    ) -> List[Union[GradioMessage, BaseMessage]]:
        """Convert chat history between formats"""
        if not history:
            return []
            
        converted_history = []
        
        for message in history:
            if to_format == "gradio":
                if isinstance(message, BaseMessage):
                    converted_history.append(self.langchain_to_gradio(message))
                elif isinstance(message, GradioMessage):
                    converted_history.append(message)
                else:
                    raise ValueError(f"Unsupported message type: {type(message)}")
            else:  # to_format == "langchain"
                if isinstance(message, GradioMessage):
                    converted_history.append(await self.gradio_to_langchain(message))
                elif isinstance(message, BaseMessage):
                    converted_history.append(message)
                else:
                    raise ValueError(f"Unsupported message type: {type(message)}")
                    
        return converted_history
        
    async def process_message(
        chat_assistant,
        message: GradioMessage,
        history: List[GradioMessage],
        model_choice: str,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None,
        history_flag: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_context: bool = True
    ) -> AsyncGenerator[GradioMessage, None]:
        """Process messages with proper format handling"""
        try:
            # Convert current message to LangChain format
            langchain_message = await self.gradio_to_langchain(message)
            
            # Convert history if needed
            langchain_history = []
            if history_flag and history:
                langchain_history = await self.convert_history(history, to_format="langchain")
            
            # Get streaming response from chat assistant
            async for chunk in chat_assistant.chat(
                message=langchain_message,
                history=langchain_history,
                prompt_info=prompt_info,
                language_choice=language_choice,
                history_flag=history_flag,
                stream=True,
                use_context=use_context
            ):
                # Convert response chunks to Gradio format
                if isinstance(chunk, str):
                    yield GradioMessage(role="assistant", content=chunk)
                elif isinstance(chunk, BaseMessage):
                    yield self.langchain_to_gradio(chunk)
                elif isinstance(chunk, dict) and "role" in chunk and "content" in chunk:
                    yield GradioMessage(
                        role=chunk["role"],
                        content=chunk["content"]
                    )
                else:
                    yield GradioMessage(role="assistant", content=str(chunk))
                    
        except Exception as e:
            logger.error(f"Process message error: {str(e)}")
            yield GradioMessage(
                role="assistant",
                content=f"An error occurred: {str(e)}"
            )