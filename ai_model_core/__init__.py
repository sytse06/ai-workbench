# ai_model_core/__init__.py
from .shared_utils.utils import (
    get_system_prompt,
    get_prompt_template,
    _format_history,
    EnhancedContentLoader
)
from .shared_utils.factory import (
    get_model, 
    get_embedding_model
)
from .config.credentials import (
    load_credentials, 
    get_api_key
)
from .config.settings import (
    load_config,
    get_directory,
    get_prompt,
    get_prompt_list,
    update_prompt_list
)
from .model_helpers import (
    ChatAssistant,
    RAGAssistant,
    VisionAssistant,
    PromptAssistant,
    SummarizationAssistant,
    TranscriptionAssistant,
    TranscriptionContext,
    E5Embeddings
)
from .model_helpers.transcription_assistant import (
    TranscriptionError,
    FileError,
    ModelError,
    OutputError,
    AudioProcessingError
)

# Define what's available for import
__all__ = [
    # Shared Utils
    'get_system_prompt',
    'get_prompt_template',
    'format_history',
    'EnhancedContentLoader',
    'get_model', 
    'get_embedding_model',
    
    # Config
    'load_credentials',
    'get_api_key',
    'load_config',
    'get_directory',
    'get_prompt',
    'get_prompt_list',
    'update_prompt_list',
    
    # Model helpers
    'ChatAssistant',
    'RAGAssistant',
    'VisionAssistant',
    'PromptAssistant',
    'SummarizationAssistant',
    'TranscriptionAssistant',
    'TranscriptionContext',
    'E5Embeddings',
    
    # Error handling
    'TranscriptionError',
    'FileError',
    'ModelError',
    'OutputError',
    'AudioProcessingError'
]