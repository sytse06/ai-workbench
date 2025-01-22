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

# Create a list of actually imported items
__all__ = [
    'ChatAssistant',
    'RAGAssistant',
    'SummarizationAssistant',
    'TranscriptionAssistant',
    'TranscriptionContext',
    'E5Embeddings',
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