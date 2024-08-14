# ai_model_interface/__init__.py
from .factory import get_model
from .utils import format_prompt
from .models.ollama import OllamaModel, OllamaRunnable