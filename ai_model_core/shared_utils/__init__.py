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
    WHISPER_MODELS,
    OUTPUT_FORMATS
)
from .factory import ModelType

# Initialize the model registry when the module is imported
initialize_model_registry()

__all__ = [
    'EnhancedContentLoader',
    'BaseMessageProcessor',
    'MessageProcessor',
    'GradioMessage',
    'GradioContent',
    'GradioFileContent',
    'GradioRole',
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
    'OUTPUT_FORMATS'
    'load_credentials', 
    'get_api_key',
    'load_config',
    'get_directory'
]