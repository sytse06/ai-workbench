# ai_model_core/config/credentials.py
import os
import json
import logging

logger = logging.getLogger(__name__)

def load_credentials():
    cred_path = os.getenv('CREDENTIALS_PATH', '../../../credentials.json')
    
    try:
        with open(cred_path, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
        os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']
        os.environ['ANTHROPIC_API_KEY'] = credentials['anthropic_api_key']
        logger.info("Credentials loaded successfully")
    except FileNotFoundError:
        logger.error(f"Credentials file not found at {cred_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in credentials file at {cred_path}")
        raise
    except KeyError as e:
        logger.error(f"Key {str(e)} not found in credentials file")
        raise
    
def get_api_key(provider: str) -> str:
    if provider.lower() == 'openai':
        key = os.getenv('OPENAI_API_KEY')
        logger.debug("Accessed OpenAI API key.")  # Avoid logging the actual key
        return key
    elif provider.lower() == 'anthropic':
        key = os.getenv('ANTHROPIC_API_KEY')
        logger.debug("Accessed Anthropic API key.")  # Avoid logging the actual key
        return key
    else:
        raise ValueError(f"Unknown provider: {provider}")