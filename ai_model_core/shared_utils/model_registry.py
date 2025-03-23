# ai_model_core/shared_utils/model_registry.py
from .factory import ModelFactory, ModelType, ModelConfig
from .model_provider import (
    ollama_provider,
    openai_provider,
    mistral_provider,
    google_provider,
    whisper_provider,
    custom_embedding_provider,
    anthropic_provider,
    cohere_provider
)

# Register model providers
def initialize_model_registry():
    """Initialize the model registry with all available providers and models"""
    
    ModelFactory.register_provider("ollama", ollama_provider)
    ModelFactory.register_provider("openai", openai_provider)
    ModelFactory.register_provider("anthropic", anthropic_provider)
    ModelFactory.register_provider("cohere", cohere_provider)
    ModelFactory.register_provider("mistral", mistral_provider)
    ModelFactory.register_provider("google", google_provider)
    ModelFactory.register_provider("whisper", whisper_provider)
    ModelFactory.register_provider("custom", custom_embedding_provider)
    
    # Register Ollama chat models
    ollama_provider.register_model(ModelConfig(
        name="Ollama (llama3.2)",
        model_id="llama3.2:latest",
        provider="ollama",
        type=ModelType.CHAT,
        description="Ollama Llama 3.2 model"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="Ollama (llama3.2-vision)",
        model_id="llama3.2-vision",
        provider="ollama",
        type=ModelType.VISION,
        description="Ollama Llama 3.2 with vision capabilities"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="Ollama (phi4)",
        model_id="phi4:latest",
        provider="ollama",
        type=ModelType.CHAT,
        description="Ollama Phi-4 model"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="Ollama (qwen2.5:14b)",
        model_id="qwen2.5:14b",
        provider="ollama",
        type=ModelType.CHAT,
        description="Ollama Qwen 2.5 14B model"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="Ollama (llava)",
        model_id="llava:latest",
        provider="ollama",
        type=ModelType.VISION,
        description="Ollama LLaVA model with vision capabilities"
    ))
    
    # Register Ollama embedding models
    ollama_provider.register_model(ModelConfig(
        name="nomic-embed-text",
        model_id="nomic-embed-text",
        provider="ollama",
        type=ModelType.EMBEDDING,
        description="Nomic text embedding model via Ollama"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="bge-large",
        model_id="bge-large",
        provider="ollama",
        type=ModelType.EMBEDDING,
        description="BGE Large embedding model via Ollama"
    ))
    
    ollama_provider.register_model(ModelConfig(
        name="bge-m3",
        model_id="bge-m3",
        provider="ollama",
        type=ModelType.EMBEDDING,
        description="BGE M3 embedding model via Ollama"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="Cohere Command",
        model_id="command",
        provider="cohere",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere Command - general purpose model"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="Cohere Command R",
        model_id="command-r",
        provider="cohere",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere Command R - model with reasoning abilities"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="Cohere Command R+",
        model_id="command-r-plus",
        provider="cohere",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere Command R+ - enhanced reasoning model"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="Cohere Command Light",
        model_id="command-light",
        provider="cohere",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere Command Light - lightweight model"
    ))
    
    # Register Cohere embedding models
    cohere_provider.register_model(ModelConfig(
        name="cohere-embed-english-v3.0",
        model_id="embed-english-v3.0",
        provider="cohere",
        type=ModelType.EMBEDDING,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere English embedding model v3.0"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="cohere-embed-multilingual-v3.0",
        model_id="embed-multilingual-v3.0",
        provider="cohere",
        type=ModelType.EMBEDDING,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere multilingual embedding model v3.0"
    ))
    
    # Register Cohere reranker model
    cohere_provider.register_model(ModelConfig(
        name="cohere-rerank-english-v3.0",
        model_id="rerank-english-v3.0",
        provider="cohere",
        type=ModelType.RERANKER,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere English reranker v3.0"
    ))
    
    cohere_provider.register_model(ModelConfig(
        name="cohere-rerank-multilingual-v3.0",
        model_id="rerank-multilingual-v3.0",
        provider="cohere",
        type=ModelType.RERANKER,
        requires_api_key=True,
        api_key_name="cohere",
        description="Cohere multilingual reranker v3.0"
    ))
        
    # Register OpenAI models
    openai_provider.register_model(ModelConfig(
        name="OpenAI o3-mini",
        model_id="o3-mini",
        provider="openai",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openai",
        description="OpenAI O3-mini model"
    ))
    
    openai_provider.register_model(ModelConfig(
        name="OpenAI GPT-4o-mini",
        model_id="gpt-4o-mini",
        provider="openai",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openai",
        description="OpenAI GPT-4o-mini model"
    ))
    
    openai_provider.register_model(ModelConfig(
        name="OpenAI GPT-4o",
        model_id="gpt-4o",
        provider="openai",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openai",
        description="OpenAI GPT-4o model"
    ))
    
    # Register OpenRouter-based models (using OpenAI provider with different base_url)
    openai_provider.register_model(ModelConfig(
        name="Deepseek v3",
        model_id="deepseek/deepseek-chat",
        provider="openai",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openrouter",
        default_params={
            "base_url": "https://openrouter.ai/api/v1",
            "x_title": "ai-workbench"
        },
        description="Deepseek Chat v3 via OpenRouter"
    ))
        
    # Register Anthropic models for direct API access
    anthropic_provider.register_model(ModelConfig(
        name="Claude 3 Opus",
        model_id="claude-3-opus-20240229",
        provider="anthropic",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="anthropic",
        description="Claude 3 Opus - most capable Claude model"
    ))
    
    anthropic_provider.register_model(ModelConfig(
        name="Claude 3 Sonnet",
        model_id="claude-3-sonnet-20240229",
        provider="anthropic",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="anthropic",
        description="Claude 3 Sonnet - balanced performance and efficiency"
    ))
    
    anthropic_provider.register_model(ModelConfig(
        name="Claude 3 Haiku",
        model_id="claude-3-haiku-20240307",
        provider="anthropic",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="anthropic",
        description="Claude 3 Haiku - fastest and most compact Claude model"
    ))
    
    anthropic_provider.register_model(ModelConfig(
        name="Claude 3.5 Sonnet",
        model_id="claude-3-5-sonnet-20240620",
        provider="anthropic",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="anthropic",
        description="Claude 3.5 Sonnet - latest Claude model with enhanced capabilities"
    ))
    
    # Also keep the existing OpenRouter registrations
    openai_provider.register_model(ModelConfig(
        name="Claude Sonnet (via OpenRouter)",
        model_id="anthropic/claude-3.5-sonnet",
        provider="openrouter",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openrouter",
        default_params={
            "x_title": "ai-workbench"
        },
        description="Claude 3.5 Sonnet via OpenRouter"
    ))
    
    openai_provider.register_model(ModelConfig(
        name="Claude Sonnet beta (via OpenRouter)",
        model_id="anthropic/claude-3.5-sonnet:beta",
        provider="openrouter",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="openrouter",
        default_params={
            "x_title": "ai-workbench"
        },
        description="Claude 3.5 Sonnet beta via OpenRouter"
    ))
    
    # Register Mistral models
    mistral_provider.register_model(ModelConfig(
        name="Mistral (large)",
        model_id="mistral-large-latest",
        provider="mistral",  # Now using dedicated provider
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="mistral",
        description="Mistral Large (latest)"
    ))
    
    mistral_provider.register_model(ModelConfig(
        name="Mistral (small)",
        model_id="mistral-small-latest",
        provider="mistral",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="mistral",
        default_params={
            "base_url": "https://api.mistral.ai/v1"
        },
        description="Mistral Small (latest)"
    ))
    
    mistral_provider.register_model(ModelConfig(
        name="Mistral (pixtral)",
        model_id="pixtral-12b-2409",
        provider="mistral",
        type=ModelType.VISION,
        requires_api_key=True,
        api_key_name="mistral",
        default_params={
            "base_url": "https://api.mistral.ai/v1"
        },
        description="Mistral Pixtral (12B) with vision capabilities"
    ))
    
    # Register Google models
    google_provider.register_model(ModelConfig(
        name="Gemini 1.5 flash",
        model_id="gemini-1.5-flash",
        provider="google",
        type=ModelType.CHAT,
        requires_api_key=True,
        api_key_name="google",
        description="Google Gemini 1.5 flash model"
    ))
    
    # Register Whisper models
    for size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]:
        whisper_provider.register_model(ModelConfig(
            name=f"Whisper {size}",
            model_id=size,
            provider="whisper",
            type=ModelType.TRANSCRIPTION,
            description=f"Whisper {size} transcription model"
        ))
    
    # Register custom embedding models
    for model_name in ["e5-base", "e5-large"]:
        custom_embedding_provider.register_model(ModelConfig(
            name=model_name,
            model_id=f"intfloat/multilingual-{model_name}",
            provider="custom",
            type=ModelType.EMBEDDING,
            description=f"E5 {model_name.split('-')[1]} embedding model"
        ))
    
    # Register OpenAI embedding models
    for model_id in ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]:
        openai_provider.register_model(ModelConfig(
            name=model_id,
            model_id=model_id,
            provider="openai",
            type=ModelType.EMBEDDING,
            requires_api_key=True,
            api_key_name="openai",
            description=f"OpenAI {model_id} embedding model"
        ))

# Export convenient functions with the same interface as before
def get_model(model_name: str, **kwargs):
    """Get a language model by name"""
    return ModelFactory.get_model(model_name, **kwargs)

def get_embedding_model(model_name: str, **kwargs):
    """Get an embedding model by name"""
    return ModelFactory.get_model(model_name, **kwargs)

async def update_model(new_model_name: str, current_model_name: str):
    """Update model if needed"""
    return await ModelFactory.update_model(new_model_name, current_model_name)

def get_reranker(model_name: str, **kwargs):
    """Get a reranker model by name"""
    return ModelFactory.get_model(model_name, **kwargs)

# Constants for backward compatibility
WHISPER_MODELS = ModelFactory.list_models(ModelType.TRANSCRIPTION)
OUTPUT_FORMATS = ["none", "txt", "srt", "vtt", "tsv", "json", "all"]