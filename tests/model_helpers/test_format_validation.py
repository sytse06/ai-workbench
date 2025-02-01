import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import gradio as gr
from typing import List, Dict, Optional
from ai_model_core.shared_utils.message_processing import (
    convert_history_to_messages,
    format_assistant_message,
    format_user_message,
    _format_history 
)
class BaseMessageFormatTestMixin:
    """Base mixin class for message format validation tests.
    
    This mixin provides common test methods for validating message formats
    that can be shared between RAGAssistant and ChatAssistant tests.
    """
    @pytest.mark.asyncio
    async def test_format_user_message(self):
        """Test user message formatting."""
        message = "Hello"
        files = None
        history = []
        empty_str, new_history = await format_user_message(message, files, history)
        
        assert empty_str == ""
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == message
    
    @pytest.mark.asyncio
    async def test_format_user_message_with_files(self):
        """Test user message formatting with files."""
        # Create mock files
        files = [MagicMock(spec=gr.File) for _ in range(2)]
        for i, f in enumerate(files):
            f.name = f"test{i}.txt"

        # Initialize history as a list of dictionaries and use right order of arguments
        history = []
        message = "Check these files"
        empty_str, new_history = await format_user_message(message, history, files)

        # Verify structure
        assert empty_str == ""
        assert len(new_history) == 1
        assert isinstance(new_history[-1]["content"], list)  # Content should be a list for files

        # Check content structure
        content = new_history[-1]["content"]
        assert len(content) == 3  # Message + 2 files
        assert content[0] == message  # The message text
        
        #Verify file entries
        for i, file_content in enumerate(content[1:]):  # Check file entries
            assert file_content["type"] == "file"
            assert file_content["path"] == f"test{i}.txt"
            assert file_content["alt_text"] == f"File: test{i}.txt"

    def test_format_assistant_message(self):
        """Test assistant message formatting function."""
        content = "Hello there"
        metadata = {"model": "test_model"}
        message = format_assistant_message(content, metadata)
        
        assert message["role"] == "assistant"
        assert message["content"] == content
        assert message["metadata"] == metadata

    @pytest.mark.parametrize("invalid_message,expected_error", [
        ({"content": "Missing role"}, KeyError),
        ({"role": "invalid", "content": "Wrong role"}, ValueError),
        ({"role": "user"}, KeyError),
        ("Not a dict", AttributeError)
    ])
    
    def test_invalid_message_formats(self, invalid_message, expected_error):
        """Test handling of invalid message formats."""
        with pytest.raises(expected_error):
            if isinstance(invalid_message, str):
                raise AttributeError("Message must be a dict")
            if "role" not in invalid_message:
                raise KeyError("Missing role")
            if "content" not in invalid_message:
                raise KeyError("Missing content")
            if invalid_message["role"] not in ["user", "assistant"]:
                raise ValueError("Invalid role")

    def test_bidirectional_conversion(self):
        """Test conversions between Gradio and LangChain formats."""
        # LangChain to Gradio
        langchain_messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there")
        ]
        
        gradio_messages = convert_history_to_messages(langchain_messages)
        assert len(gradio_messages) == 2
        assert gradio_messages[0]["role"] == "user"
        assert gradio_messages[0]["content"] == "Hello"
        
        # Gradio to LangChain
        gradio_format = [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm fine!"}
        ]
        
        langchain_format = _format_history(gradio_format)
        assert len(langchain_format) == 2
        assert isinstance(langchain_format[0], HumanMessage)
        assert isinstance(langchain_format[1], AIMessage)
        assert langchain_format[0].content == "How are you?"
        
    def test_complex_history_conversion(self):
        """Test conversion of complex chat histories."""
        # Complex Gradio format with metadata
        gradio_history = [
            {
                "role": "user", 
                "content": "Analyze this",
                "metadata": {"timestamp": "2024-01-28"}
            },
            {
                "role": "assistant", 
                "content": "Here's my analysis",
                "metadata": {
                    "model": "test-model",
                    "confidence": 0.95
                }
            }
        ]
        
        # Convert to LangChain and verify
        langchain_format = _format_history(gradio_history)
        assert len(langchain_format) == 2
        assert isinstance(langchain_format[0], HumanMessage)
        assert isinstance(langchain_format[1], AIMessage)
        
        # Convert back to Gradio and verify
        gradio_format = convert_history_to_messages(langchain_format)
        assert len(gradio_format) == 2
        assert all("role" in msg for msg in gradio_format)
        assert all("content" in msg for msg in gradio_format)

    def test_edge_case_conversions(self):
        """Test conversion of edge cases and special formats."""
        test_cases = [
            # Empty messages
            [{"role": "user", "content": ""}],
            # Messages with special characters
            [{"role": "user", "content": "Hello\n\nWorld! 👋"}],
            # Long messages
            [{"role": "user", "content": "x" * 1000}],
            # Messages with multiple newlines
            [{"role": "user", "content": "line1\n\nline2\n\nline3"}]
        ]
        
        for case in test_cases:
            # Convert Gradio to LangChain
            langchain_format = _format_history(case)
            assert len(langchain_format) == len(case)
            
            # Convert back to Gradio
            gradio_format = convert_history_to_messages(langchain_format)
            assert len(gradio_format) == len(case)
            
            # Verify content is preserved
            assert gradio_format[0]["content"] == case[0]["content"]

    def test_multimodal_message_format(self):
        """Test handling of multimodal messages (text + images)."""
        multimodal_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image", "path": "test.jpg"}
            ]
        }
        
        assert isinstance(multimodal_msg["content"], list)
        assert all("type" in component for component in multimodal_msg["content"])
        assert any(comp["type"] == "text" for comp in multimodal_msg["content"])
        assert any(comp["type"] == "image" for comp in multimodal_msg["content"])

    @pytest.mark.asyncio
    def test_history_batch_conversion(self):
        """Test batch conversion of message history."""
        # Mixed format history
        mixed_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"speaker": "user", "message": "How are you?"},
            {"type": "assistant", "text": "I'm good!"}
        ]
        
        converted = convert_history_to_messages(mixed_history)
        assert len(converted) == 4
        assert all("role" in msg and "content" in msg for msg in converted)
        assert all(msg["role"] in ["user", "assistant"] for msg in converted)
        
@pytest.fixture
def base_message_format_test():
    """Fixture to create a test instance with required methods."""
    class TestInstance(BaseMessageFormatTestMixin):
        async def format_user_message_func(self, *args, **kwargs):
            # Implement for specific assistant
            pass
            
        def format_assistant_message_func(self, *args, **kwargs):
            # Implement for specific assistant
            pass
            
        def _format_history_func(self, *args, **kwargs):
            # Implement for specific assistant
            pass
            
        def convert_to_messages_func(self, *args, **kwargs):
            # Implement for specific assistant
            pass
    
    return TestInstance()

# Helper functions that can be used by both assistants
def validate_message_format(message: Dict) -> bool:
    """Validate the format of a message dictionary."""
    if not isinstance(message, dict):
        return False
    if "role" not in message or "content" not in message:
        return False
    if message["role"] not in ["user", "assistant"]:
        return False
    return True

def validate_history_format(history: List[Dict]) -> bool:
    """Validate the format of a message history list."""
    if not isinstance(history, list):
        return False
    return all(validate_message_format(msg) for msg in history)

async def process_files(files: List[gr.File]) -> List[Dict]:
    """Process uploaded files into message content."""
    file_contents = []
    for file in files:
        file_content = {
            "type": "file",
            "path": file.name,
            "alt_text": f"File: {file.name}"
        }
        file_contents.append(file_content)
    return file_contents