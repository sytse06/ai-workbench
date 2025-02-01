# ai_model_core/shared_utils/__init__.py
from .utils import EnhancedContentLoader

from .message_processing import (
    format_user_message,
    format_assistant_message,
    format_file_content,
    convert_history_to_messages,
    _format_history,
    process_message
)

from .prompt_utils import (
    get_prompt, 
    get_prompt_list,
    update_prompt_list,
    get_prompt_template, 
    get_system_prompt
)

from .factory import (
    get_model,
    update_model, 
    get_embedding_model,
    WHISPER_MODELS,
    OUTPUT_FORMATS
)
from ..config.credentials import (
    load_credentials, 
    get_api_key
)
from ..config.settings import (
    load_config,
    get_directory
)

__all__ = [
    format_user_message,
    format_assistant_message,
    format_file_content,
    convert_history_to_messages,
    _format_history,
    process_message,
    get_prompt, 
    get_prompt_list,
    update_prompt_list,
    get_prompt_template, 
    get_system_prompt,
    get_model,
    update_model, 
    get_embedding_model,
    load_credentials, 
    get_api_key,
    load_config,
    get_directory
]