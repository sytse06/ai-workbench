"""Model helpers package initialization."""

# Define __all__ for what should be available when importing *
__all__ = [
    'E5Embeddings',
    'ChatAssistant',
    'RAGAssistant',
    'PromptAssistant',
    'SummarizationAssistant',
    'TranscriptionAssistant',
    'TranscriptionContext',
    'VisionAssistant'
]


def __getattr__(name):
    """Lazy import implementation."""
    if name in __all__:
        if name == 'E5Embeddings':
            from .embeddings import E5Embeddings
            return E5Embeddings
        elif name == 'ChatAssistant':
            from .chat_assistant import ChatAssistant
            return ChatAssistant
        elif name == 'RAGAssistant':
            from .RAG_assistant import RAGAssistant
            return RAGAssistant
        elif name == 'PromptAssistant':
            from .prompt_assistant import PromptAssistant
            return PromptAssistant
        elif name == 'SummarizationAssistant':
            from .summarize_assistant import SummarizationAssistant
            return SummarizationAssistant
        elif name == 'TranscriptionAssistant':
            from .transcription_assistant import TranscriptionAssistant
            return TranscriptionAssistant
        elif name == 'TranscriptionContext':
            from .transcription_assistant import TranscriptionContext
            return TranscriptionContext
        elif name == 'VisionAssistant':
            from .vision_assistant import VisionAssistant
            return VisionAssistant
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
