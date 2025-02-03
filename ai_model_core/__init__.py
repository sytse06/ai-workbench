# ai_model_core/__init__.py
from .shared_utils.utils import EnhancedContentLoader
from .shared_utils.message_processing import (
    format_user_message,
    format_assistant_message,
    format_system_message,
    convert_gradio_to_langchain,
    convert_langchain_to_gradio,
    convert_history,
    process_message,
)
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

# Define everything that should be available for import
__all__ = [
    # Core functionality
    'EnhancedContentLoader',
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
    'WHISPER_MODELS',
    'OUTPUT_FORMATS',
    
    # Message processing
    'format_user_message',
    'format_assistant_message',
    'format_file_content',
    'convert_history_to_messages',
    '_format_history',
    'process_message',
    
    # Prompt handling
    'get_prompt',
    'get_prompt_list',
    'update_prompt_list',
    'get_prompt_template',
    'get_system_prompt',
    
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