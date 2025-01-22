# ai_model_core/__init__.py
from .shared_utils.utils import (
    get_system_prompt,
    get_prompt_template,
    get_prompt,
    get_prompt_list,
    update_prompt_list,
    _format_history,
    EnhancedContentLoader,
    format_assistant_message,
    format_user_message,
    format_file_content,
    convert_history_to_messages
)
from .shared_utils.factory import (
    get_model,
    update_model, 
    get_embedding_model
)
from .config.credentials import (
    load_credentials, 
    get_api_key
)
from .config.settings import (
    load_config,
    get_directory
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
    'get_prompt',
    'get_prompt_list',
    'update_prompt_list',
    '_format_history',
    'EnhancedContentLoader',
    'get_model',
    'update_model', 
    'get_embedding_model',
    
    # Config
    'load_credentials',
    'get_api_key',
    'load_config',
    'get_directory',
    
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