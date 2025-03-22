# ai_model_core/__init__.py
from .shared_utils.content_loader import EnhancedContentLoader
from .shared_utils.content_processor import (
    BaseContentProcessor,
    ChatContentProcessor,
    RAGContentProcessor, 
    SummarizationContentProcessor,
    TranscriptionContentProcessor,
    AssistantType,
    LoaderConfig
)
from .shared_utils.content_coordinator import (
    ContentProcessingComponent,
    setup_content_processing
)
from .shared_utils.message_types import (
    BaseMessageProcessor,
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from .shared_utils.message_processing import MessageProcessor
from .shared_utils.prompt_utils import (
    get_prompt, 
    get_prompt_list,
    update_prompt_list,
    get_prompt_template, 
    get_system_prompt
)
from .shared_utils.factory import (
    get_model,
    update_model, 
    get_embedding_model,
    get_reranker,
    ModelType 
)
from .shared_utils.model_registry import (
    WHISPER_MODELS,
    OUTPUT_FORMATS
)
from .config.credentials import load_credentials, get_api_key
from .config.settings import load_config, get_directory
from .model_helpers.chat_assistant import ChatAssistant
from .model_helpers.RAG_assistant import RAGAssistant
from .model_helpers.summarize_assistant import SummarizationAssistant
from .model_helpers.transcription_assistant import (
    TranscriptionAssistant,
    TranscriptionContext,
    TranscriptionError,
    FileError,
    ModelError,
    OutputError,
    AudioProcessingError
)
from .model_helpers.embeddings import E5Embeddings

# Definitions for imports elsewhere
__all__ = [
    # Content loading and processing
    'EnhancedContentLoader',
    'BaseContentProcessor',
    'ChatContentProcessor',
    'RAGContentProcessor',
    'SummarizationContentProcessor',
    'TranscriptionContentProcessor',
    'ContentProcessingComponent',
    'setup_content_processing',
    'AssistantType',
    'LoaderConfig',
    
    # Message processing
    'BaseMessageProcessor',
    'MessageProcessor',
    'GradioMessage',
    'GradioContent',
    'GradioFileContent',
    'GradioRole',
    
    # Assistants
    'ChatAssistant',
    'RAGAssistant',
    'VisionAssistant',
    'PromptAssistant',
    'SummarizationAssistant',
    'TranscriptionAssistant',
    'TranscriptionContext',
    'E5Embeddings',
    
    # Utility functions
    'get_model',
    'update_model',
    'get_embedding_model',
    'get_reranker',
    'ModelType',
    'WHISPER_MODELS',
    'OUTPUT_FORMATS',
    
    # Configuration
    'load_credentials',
    'get_api_key',
    'load_config',
    'get_directory',
    
    # Error handling
    'TranscriptionError',
    'FileError',
    'ModelError',
    'OutputError',
    'AudioProcessingError'
]