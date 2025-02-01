import sys
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from langchain.schema import AIMessage, Document
from ai_model_core.model_helpers.chat_assistant import (
    ChatAssistant
)
from ai_model_core.shared_utils.utils import (
    EnhancedContentLoader
)

# Add project root to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

@pytest.fixture(autouse=True)
def setup_testing_env():
    """Setup testing environment variables."""
    os.environ["TESTING"] = "true"
    yield
    os.environ.pop("TESTING", None)
    
@pytest.fixture
def test_input_dir():
    """Create and manage input directory for tests."""
    input_dir = Path("input/tmp")
    input_dir.mkdir(parents=True, exist_ok=True)
    yield input_dir
    # Clean up files but keep directory
    for file in input_dir.glob("*"):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Warning: Could not delete {file}: {e}")

class MockStreamingModel:
    """Mock chat model for testing."""
    def __init__(self, response="Test response"):
        self.response = response
        
    def bind(self, **kwargs):
        """Mock bind method."""
        return self
        
    async def astream(self, messages, **kwargs):
        """Mock streaming response."""
        yield AIMessage(content=self.response)
        
    async def agenerate(self, messages, **kwargs):
        """Mock non-streaming response."""
        return MockLLMResult(generations=[[MockGeneration(text=self.response)]])

class MockLLMResult:
    def __init__(self, generations):
        self.generations = generations

class MockGeneration:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def chat_assistant(test_input_dir):
    """Provides a ChatAssistant instance with test model."""
    assistant = ChatAssistant(
        model_choice="test_model",
        temperature=0.7,
        max_tokens=500,
        temp_dir=str(test_input_dir)
    )
    # Mock the content loader for file handling tests
    assistant.content_loader = MagicMock(spec=EnhancedContentLoader)
    assistant.content_loader.load_documents.return_value = [
        Document(page_content="test content", metadata={"source": "test.txt"})
    ]
    return assistant

@pytest.fixture
def mock_chat_model():
    """Provides a mock chat model with basic streaming capability."""
    model = Mock()
    model.astream = AsyncMock(return_value=[AIMessage(content="Test response")])
    return model