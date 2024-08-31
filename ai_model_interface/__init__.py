# ai_model_interface/__init__.py
from .factory import get_model
from .utils import get_system_prompt, get_prompt_template, _format_history
from .config.credentials import load_credentials, get_api_key
from .config.settings import load_config, get_directory, get_prompt, get_prompt_list, update_prompt_list
from .model_helpers import VisionAssistant, convert_to_base64, image_chat

__all__ = [
    'get_model', 'format_prompt', 'get_system_prompt', 'get_prompt_template', 'format_history',
    'load_credentials', 'get_api_key', 'load_config', 'get_directory', 'get_prompt',
    'get_prompt_list', 'update_prompt_list', 'VisionAssistant'
]