import sys
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from langchain.schema import AIMessage
from ai_model_core.model_helpers.chat_assistant import ChatAssistant

# Add project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

@pytest.fixture
def mock_chat_model():
    """Provides a mock chat model with basic streaming capability."""
    model = Mock()
    model.astream = AsyncMock(return_value=[AIMessage(content="Test response")])
    return model

@pytest.fixture
def chat_assistant(mock_chat_model):
    """Provides a ChatAssistant instance with mocked model."""
    with patch('ai_model_core.shared_utils.factory.get_model', 
               return_value=mock_chat_model):
        assistant = ChatAssistant(
            model_choice="test_model",
            temperature=0.7,
            max_tokens=500
        )
        return assistant