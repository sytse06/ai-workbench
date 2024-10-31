# ai_model_core/__init__.py
# Base utilities and configurations
from .utils import (
    get_system_prompt,
    get_prompt_template,
    _format_history,
    EnhancedContentLoader
)
from .config.credentials import load_credentials, get_api_key
from .config.settings import (
    load_config,
    get_directory,
    get_prompt,
    get_prompt_list,
    update_prompt_list
)

# Define what's available for import
__all__ = [
    # Utils
    'get_system_prompt',
    'get_prompt_template',
    'format_history',
    'EnhancedContentLoader',
    
    # Config
    'load_credentials',
    'get_api_key',
    'load_config',
    'get_directory',
    'get_prompt',
    'get_prompt_list',
    'update_prompt_list',
]

# Lazy loading for model-related components
def get_model(*args, **kwargs):
    from .factory import get_model
    return get_model(*args, **kwargs)

def get_embedding_model(*args, **kwargs):
    from .factory import get_embedding_model
    return get_embedding_model(*args, **kwargs)

# Add factory functions to __all__
__all__ += ['get_model', 'get_embedding_model']

# Lazy loading for assistants
def load_assistants():
    from .model_helpers import (
        ChatAssistant,
        RAGAssistant,
        VisionAssistant,
        PromptAssistant,
        SummarizationAssistant,
        TranscriptionAssistant
    )
    from .model_helpers.RAG_assistant import E5Embeddings
    
    global ChatAssistant, RAGAssistant, VisionAssistant, PromptAssistant
    global SummarizationAssistant, TranscriptionAssistant, TranscriptionContext, E5Embeddings
    
    # Add assistants to __all__
    __all__.extend([
        'ChatAssistant',
        'RAGAssistant',
        'VisionAssistant',
        'PromptAssistant',
        'SummarizationAssistant',
        'TranscriptionAssistant',
        'TranscriptionContext',
        'E5Embeddings'
    ])

# Load assistants when needed
load_assistants()