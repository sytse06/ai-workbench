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

# Third-party imports
from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    BaseMessage
)

# Type definitions for Gradio message structure
GradioRole = Literal["user", "assistant", "system"]
GradioFileContent = Dict[str, str]
GradioContent = Union[str, GradioFileContent, List[Union[str, GradioFileContent]]]

@dataclass
class GradioMessage:
    """Type-safe representation of a Gradio message"""
    role: GradioRole
    content: GradioContent
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GradioMessage':
        """Create GradioMessage from dictionary format"""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", "") or data.get("text", "")
        )

class BaseMessageProcessor:
    """Base class defining the interface for message processing"""
    
    def format_user_message(
        self,
        message: Union[str, Dict[str, Any]], 
        files: Optional[List[Any]] = None
    ) -> GradioMessage:
        """Format raw user input into a GradioMessage"""
        raise NotImplementedError
        
    def format_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> GradioMessage:
        """Format assistant response into a GradioMessage"""
        raise NotImplementedError
        
    def format_system_message(
        self, 
        content: str
    ) -> GradioMessage:
        """Format system message into a GradioMessage"""
        raise NotImplementedError

    async def process_message_content(
        self,
        content: GradioContent
    ) -> str:
        """Process Gradio content into LangChain-compatible string"""
        raise NotImplementedError

    async def gradio_to_langchain(
        self,
        message: GradioMessage
    ) -> BaseMessage:
        """Convert Gradio message format to LangChain message"""
        raise NotImplementedError

    def langchain_to_gradio(
        self,
        message: BaseMessage
    ) -> GradioMessage:
        """Convert LangChain message to Gradio format (sync as it's just data transformation)"""
        raise NotImplementedError
        
    async def convert_history(
        self,
        history: List[Union[GradioMessage, BaseMessage]],
        to_format: Literal["gradio", "langchain"] = "gradio"
    ) -> List[Union[GradioMessage, BaseMessage]]:
        """Convert chat history between different formats"""
        raise NotImplementedError
        
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
        raise NotImplementedError