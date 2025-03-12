# ai_model_core/shared_utils/message_types.py
# Standard library imports
from typing import (
    List, 
    Dict, 
    Any, 
    Union, 
    Optional,
    AsyncGenerator,
    Literal
)
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third-party imports
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)

# Type definitions for Gradio message structure
GradioRole = Literal["user", "assistant", "system"]
GradioFileContent = Dict[str, str]  # {"path": "/path/to/file.jpg"}
GradioContent = Union[str, GradioFileContent, List[Union[str, GradioFileContent]]]

@dataclass
class GradioMessage:
    """Representation of a Gradio message format"""
    role: GradioRole
    content: GradioContent
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradioMessage':
        """Create GradioMessage from dictionary format"""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", "") or data.get("text", "")
        )

class BaseMessageProcessor(ABC):
    """Abstract base class for message processing operations"""

    @abstractmethod    
    def format_user_message(
        self,
        message: Union[str, Dict[str, Any]], 
        files: Optional[List[Any]] = None
    ) -> GradioMessage:
        """Format raw user input into a GradioMessage"""
        pass
    
    @abstractmethod    
    def format_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> GradioMessage:
        """Format assistant response into a GradioMessage"""
        pass

    @abstractmethod        
    def format_system_message(
        self, 
        content: str
    ) -> GradioMessage:
        """Format system message into a GradioMessage"""
        pass

    @abstractmethod
    async def process_message_content(
        self,
        content: GradioContent
    ) -> str:
        """Process Gradio content into LangChain-compatible string"""
        pass

    @abstractmethod
    async def gradio_to_langchain(
        self,
        message: GradioMessage
    ) -> BaseMessage:
        """Convert Gradio message format to LangChain message"""
        pass

    @abstractmethod
    def langchain_to_gradio(
        self,
        message: BaseMessage
    ) -> GradioMessage:
        """Convert LangChain message to Gradio format (sync as it's just data transformation)"""
        pass

    @abstractmethod        
    async def convert_history(
        self,
        history: List[Union[GradioMessage, BaseMessage]],
        to_format: Literal["gradio", "langchain"] = "gradio"
    ) -> List[Union[GradioMessage, BaseMessage]]:
        """Convert chat history between different formats"""
        pass

    @abstractmethod        
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
        """Process messages ensuring proper format for both Gradio and LangChain"""
        pass
    
    @abstractmethod
    async def get_message_text(
        self, 
        message: Union[str, Dict, GradioMessage, BaseMessage]
    ) -> str:
        """Extract plain text from various message formats"""
        pass