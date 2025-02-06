# ai_model_core/model_helpers/__init__.py
from .chat_assistant import ChatAssistant
from .RAG_assistant import RAGAssistant
from .summarize_assistant import SummarizationAssistant
from .transcription_assistant import (
    TranscriptionAssistant,
    TranscriptionContext,
    TranscriptionError,
    FileError,
    ModelError,
    OutputError,
    AudioProcessingError
)
from .embeddings import E5Embeddings
from ..shared_utils.message_types import (
    BaseMessageProcessor,
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from ..shared_utils.message_processing import MessageProcessor

# Create a list of actually imported items
__all__ = [
    'ChatAssistant',
    'RAGAssistant',
    'SummarizationAssistant',
    'TranscriptionAssistant',
    'TranscriptionContext',
    'E5Embeddings',
    'BaseMessageProcessor',
    'MessageProcessor',
    'GradioMessage',
    'GradioContent',
    'GradioFileContent',
    'GradioRole',
    # Errors
    'TranscriptionError',
    'FileError',
    'ModelError',
    'OutputError',
    'AudioProcessingError'
]

# Optional imports - these will be None if import fails
try:
    from .prompt_assistant import PromptAssistant
    __all__.append('PromptAssistant')
except ImportError:
    PromptAssistant = None

try:
    from .vision_assistant import VisionAssistant
    __all__.append('VisionAssistant')
except ImportError:
    VisionAssistant = None