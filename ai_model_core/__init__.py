# ai_model_core/__init__.py
from .factory import get_model, get_embedding_model
from .utils import get_system_prompt, get_prompt_template, _format_history, load_document, load_web_content, split_text
from .config.credentials import load_credentials, get_api_key
from .config.settings import load_config, get_directory, get_prompt, get_prompt_list, update_prompt_list
from .model_helpers import ChatAssistant, RAGAssistant, VisionAssistant, PromptAssistant, SummarizationAssistant
from ai_model_core.model_helpers.RAG_assistant import CustomHuggingFaceEmbeddings

__all__ = [
    'get_model', 'get_embedding_model', 'format_prompt', 'get_system_prompt', 'get_prompt_template', '_format_history',
    'load_credentials', 'get_api_key', 'load_config', 'get_directory', 'get_prompt', 'load_content',
    'get_prompt_list', 'update_prompt_list', 'ChatAssistant', 'VisionAssistant', 'PromptAssistant', 
    'RAGAssistant', 'SummarizationAssistant', 'CustomHuggingFaceEmbeddings'
]