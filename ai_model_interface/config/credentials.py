# ai_model_interface/config/credentials.py
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
        return os.getenv('OPENAI_API_KEY')
    elif provider.lower() == 'anthropic':
        return os.getenv('ANTHROPIC_API_KEY')
    else:
        raise ValueError(f"Unknown provider: {provider}")