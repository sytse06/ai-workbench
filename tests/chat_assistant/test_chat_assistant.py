# tests/chat_assistant/test_chat_assistant.py
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import asyncio
from typing import List, Tuple

from langchain.schema import HumanMessage, AIMessage, Document, BaseMessage
import gradio as gr

from ai_model_core.model_helpers.chat_assistant import ChatAssistant
from ai_model_core.shared_utils.utils import EnhancedContentLoader

class MockStreamingModel:
    """Mock model that simulates LangChain streaming interface"""
    def __init__(self, response="Mock response"):
        self.response = response
    
    def bind(self, **kwargs):
        """Simulate model binding, return self to maintain chain"""
        return self
        
    async def astream(self, messages, **kwargs):
        """Simulate streaming response"""
        yield AIMessage(content=self.response)
        
    async def agenerate(self, messages, **kwargs):
        """Simulate non-streaming response"""
        return MockLLMResult(generations=[[MockGeneration(text=self.response)]])

class MockLLMResult:
    def __init__(self, generations):
        self.generations = generations

class MockGeneration:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def mock_model():
    return MockStreamingModel()

@pytest.fixture
def chat_assistant():
    with patch('ai_model_core.shared_utils.factory.get_model', return_value=MockStreamingModel()):
        assistant = ChatAssistant(
            model_choice="Ollama (LLama3.2)",
            temperature=0.7,
            max_tokens=500
        )
        # Mock the content loader
        assistant.content_loader = MagicMock(spec=EnhancedContentLoader)
        assistant.content_loader.load_documents.return_value = [
            Document(page_content="test content", metadata={"source": "test1.txt"})
        ]
        return assistant

@pytest.fixture
def mock_gradio_files():
    file1 = MagicMock(spec=gr.File)
    file1.name = str(Path("test1.txt").absolute())
    file2 = MagicMock(spec=gr.File)
    file2.name = str(Path("test2.txt").absolute())
    return [file1, file2]

class TestChatAssistant:
    @pytest.mark.asyncio
    async def test_initialization(self, chat_assistant):
        assert chat_assistant.model_choice == "Ollama (LLama3.2)"
        assert chat_assistant.temperature == 0.7
        assert chat_assistant.max_tokens == 500
        assert isinstance(chat_assistant.documents, list)

    @pytest.mark.asyncio
    async def test_update_model(self, chat_assistant):
        with patch('ai_model_core.shared_utils.factory.get_model', return_value=MockStreamingModel()):
            new_model = "Claude Sonnet"
            await chat_assistant.update_model(new_model)
            assert chat_assistant.model_choice == new_model

    @pytest.mark.asyncio
    async def test_process_chat_context_files(self, chat_assistant, mock_gradio_files):
        status, success = await chat_assistant.process_chat_context_files(mock_gradio_files)
        assert success is True
        assert "Successfully processed" in status

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message,history,history_flag", [
        ("Hello", [], True),
        ("Hello", [("Hi", "Hello!")], True),
        ("Hello", [("Hi", "Hello!")], False),
    ])
    async def test_chat_history_handling(self, chat_assistant, message, history, history_flag):
        chat_assistant.model = MockStreamingModel("Test history response")
        
        responses = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=history,
            history_flag=history_flag,
            stream=True
        ):
            responses.append(chunk)
        
        assert len(responses) > 0
        assert "Test history response" in responses[0]

    # ... (keep other existing tests) ...

@pytest.mark.asyncio
async def test_chat_wrapper():
    mock_chat_assistant = AsyncMock()
    async def mock_chat(*args, **kwargs):
        yield "Mock wrapper response"
    mock_chat_assistant.chat = mock_chat
    
    with patch('ai_model_core.model_helpers.chat_assistant.ChatAssistant', return_value=mock_chat_assistant):
        from main import chat_wrapper
        
        # Update to await the model update
        model_choice = "Ollama (LLama3.2)"
        await mock_chat_assistant.update_model(model_choice)
        
        result = []
        async for chunk in chat_wrapper(
            message="Hello",
            history=[],
            model_choice=model_choice,
            temperature=0.7,
            max_tokens=500,
            files=[],
            history_flag=True,
            use_context=True
        ):
            result.append(chunk)

        assert len(result) > 0
        assert "Mock wrapper response" in result[0]
