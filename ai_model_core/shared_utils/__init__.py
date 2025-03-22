# ai_model_core/shared_utils/__init__.py
from ..config.credentials import (
    load_credentials, 
    get_api_key
)
from ..config.settings import (
    load_config,
    get_directory
)

from .model_registry import (
    get_model,
    get_embedding_model,
    update_model,
    initialize_model_registry,
    get_reranker,
    WHISPER_MODELS,
    OUTPUT_FORMATS
)
from .factory import ModelType
from .content_loader import EnhancedContentLoader
from .content_processor import (
    BaseContentProcessor,
    ChatContentProcessor,
    RAGContentProcessor, 
    SummarizationContentProcessor,
    TranscriptionContentProcessor,
    AssistantType,
    LoaderConfig
)
from .content_coordinator import (
    ContentProcessingComponent,
    setup_content_processing
)

# Message processing components
from .message_types import (
    BaseMessageProcessor,
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from .message_processing import MessageProcessor

# Initialize the model registry when the module is imported
initialize_model_registry()

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
    
    # Prompts and models
    'get_prompt', 
    'get_prompt_list',
    'update_prompt_list',
    'get_prompt_template', 
    'get_system_prompt',
    'get_model',
    'update_model', 
    'get_embedding_model',
    'get_reranker',
    'ModelType',
    'WHISPER_MODELS',
    'OUTPUT_FORMATS',
    
    # Config
    'load_credentials', 
    'get_api_key',
    'load_config',
    'get_directory'
]