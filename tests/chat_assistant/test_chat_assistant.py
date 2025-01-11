# tests/chat_assistant/test_chat_assistant.py
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import asyncio
from typing import List, Tuple

from langchain.schema import HumanMessage, AIMessage, Document, BaseMessage
import gradio as gr

from ai_model_core.model_helpers.chat_assistant import ChatAssistant
from ai_model_core.shared_utils.utils import (
    EnhancedContentLoader,
    get_prompt_template,
    format_user_message,
    format_assistant_message,
    format_file_content,
    convert_history_to_messages
)
from ai_model_core.config.settings import (
    load_config,
    get_prompt_list,
    get_system_prompt
)

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

class TestMessageFormatting:
    def test_format_user_message(self):
        message = "Hello"
        history = []
        result, new_history = format_user_message(message, None, history)
        assert result == ""
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == message

    def test_format_user_message_with_files(self, mock_gradio_files):
        message = "Check these files"
        history = []
        result, new_history = format_user_message(message, mock_gradio_files, history)
        assert result == ""
        assert len(new_history) == 3  # 2 files + 1 message
        assert all(msg["role"] == "user" for msg in new_history)
        # Check file messages are formatted correctly
        file_messages = [msg for msg in new_history if isinstance(msg["content"], dict)]
        assert len(file_messages) == 2
        for msg in file_messages:
            assert "path" in msg["content"]
            assert "alt_text" in msg["content"]
        # Check text message is formatted correctly
        text_message = [msg for msg in new_history if isinstance(msg["content"], str)][0]
        assert text_message["content"] == message

    def test_format_assistant_message(self):
        content = "Test response"
        metadata = {"model": "test-model"}
        message = format_assistant_message(content, metadata)
        assert message["role"] == "assistant"
        assert message["content"] == content
        assert message["metadata"] == metadata

    def test_convert_history_to_messages(self):
        tuple_history = [("user msg", "assistant msg")]
        dict_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        
        # Test tuple conversion
        messages = convert_history_to_messages(tuple_history)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        
        # Test dict conversion
        messages = convert_history_to_messages(dict_history)
        assert len(messages) == 2
        assert all("role" in msg and "content" in msg for msg in messages)

class TestChatAssistant:
    @pytest.mark.asyncio
    async def test_chat_with_formatted_messages(self, chat_assistant):
        message = "Hello"
        history = [{"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "Hello!"}]
        
        _, formatted_history = format_user_message(message, None, history)
        
        responses = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=formatted_history,
            history_flag=True,
            stream=True
        ):
            responses.append(format_assistant_message(chunk))
        
        assert len(responses) > 0
        assert all(r["role"] == "assistant" for r in responses)
        assert all("content" in r for r in responses)

@pytest.fixture
def mock_model():
    return MockStreamingModel()


@pytest.fixture
def mock_config():
    return {
        "system_prompt_settings": {
            "english": {
                "system_prompt": "You are a helpful AI assistant."
            },
            "dutch": {
                "system_prompt": "Je bent een behulpzame AI-assistent."
            }
        },
        "prompts": {
            "english": ["general", "code_review", "analysis"],
            "dutch": ["algemeen", "code_review", "analyse"],
            "general": "Act as a general assistant.",
            "code_review": "Act as a code reviewer.",
            "analysis": "Act as a data analyst.",
            "algemeen": "Gedraag je als een algemene assistent.",
            "code_review": "Gedraag je als een code reviewer.",
            "analyse": "Gedraag je als een data analist."
        }
    }

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
        ("Hello", [{"role": "user", "content": "Hi"}, 
                {"role": "assistant", "content": "Hello!"}], True),
        ("Hello", [{"role": "user", "content": "Hi"}, 
                {"role": "assistant", "content": "Hello!"}], False),
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

    @pytest.mark.asyncio
    async def test_chat_with_prompt_template(self, chat_assistant, monkeypatch):
        monkeypatch.setattr('ai_model_core.config.settings.load_config', 
                           lambda: mock_config())
        
        message = "Review this code"
        history = []
        prompt_info = "code_review"
        language_choice = "english"
        
        responses = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=history,
            history_flag=True,
            stream=True,
            prompt_info=prompt_info,
            language_choice=language_choice
        ):
            responses.append(chunk)
        
        assert len(responses) > 0
        assert isinstance(responses[0], str)

    @pytest.mark.asyncio
    async def test_chat_with_prompt_template_different_languages(self, chat_assistant, monkeypatch):
        monkeypatch.setattr('ai_model_core.config.settings.load_config', 
                           lambda: mock_config())
        
        test_cases = [
            ("english", "code_review", "Review this code"),
            ("dutch", "analyse", "Analyseer deze data")
        ]
        
        for language, prompt_info, message in test_cases:
            responses = []
            async for chunk in chat_assistant.chat(
                message=message,
                history=[],
                history_flag=True,
                stream=True,
                prompt_info=prompt_info,
                language_choice=language
            ):
                responses.append(chunk)
            
            assert len(responses) > 0
            assert isinstance(responses[0], str)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prompt_info,language_choice", [
        ("general", "english"),
        ("code_review", "english"),
        ("analysis", "english"),
        ("algemeen", "dutch"),
        ("code_review", "dutch"),
        ("analyse", "dutch")
    ])
    async def test_prompt_template_variations(self, chat_assistant, prompt_info, 
                                           language_choice, monkeypatch):
        monkeypatch.setattr('ai_model_core.config.settings.load_config', 
                           lambda: mock_config())
        
        message = "Test message"
        responses = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=[],
            history_flag=True,
            stream=True,
            prompt_info=prompt_info,
            language_choice=language_choice
        ):
            responses.append(chunk)
            
        assert len(responses) > 0
        assert isinstance(responses[0], str)

    @pytest.mark.asyncio
    async def test_model_configuration(self, chat_assistant):
        # Test Ollama configuration
        chat_assistant.model_choice = "ollama"
        chat_assistant._configure_model(stream=True)
        assert chat_assistant.model.bind.called_with(stop=None, stream=True)

        # Test Gemini configuration
        chat_assistant.model_choice = "gemini"
        chat_assistant._configure_model()
        assert chat_assistant.model.bind.called_with(
            generation_config={
                "temperature": chat_assistant.temperature,
                "max_output_tokens": chat_assistant.max_tokens
            }
        )

        # Test default configuration
        chat_assistant.model_choice = "other"
        chat_assistant._configure_model(stream=True)
        assert chat_assistant.model.bind.called_with(
            temperature=chat_assistant.temperature,
            max_tokens=chat_assistant.max_tokens,
            stream=True
        )

    def test_get_document_summary(self, chat_assistant):
        # Test empty documents
        assert chat_assistant.get_document_summary() == "No documents currently loaded."

        # Test with documents
        test_docs = [
            Document(page_content="test1", metadata={"source": "file1.txt"}),
            Document(page_content="test2", metadata={"source": "file1.txt"}),
            Document(page_content="test3", metadata={"source": "file2.txt"})
        ]
        chat_assistant.documents = test_docs
        
        summary = chat_assistant.get_document_summary()
        assert "Total documents loaded: 3" in summary
        assert "file1.txt: 2 document(s)" in summary
        assert "file2.txt: 1 document(s)" in summary

    def test_clear_documents(self, chat_assistant):
        chat_assistant.documents = [
            Document(page_content="test", metadata={"source": "test.txt"})
        ]
        chat_assistant.clear_documents()
        assert len(chat_assistant.documents) == 0

    def test_set_temperature(self, chat_assistant):
        new_temp = 0.9
        chat_assistant.set_temperature(new_temp)
        assert chat_assistant.temperature == new_temp

    def test_set_max_tokens(self, chat_assistant):
        new_tokens = 1000
        chat_assistant.set_max_tokens(new_tokens)
        assert chat_assistant.max_tokens == new_tokens

    @pytest.mark.asyncio
    async def test_get_context_from_docs(self, chat_assistant):
        # Test with no context
        assert chat_assistant.get_context_from_docs("test", use_context=False) == ""
        
        # Test with context
        test_doc = Document(
            page_content="test content",
            metadata={"source": "test.txt"}
        )
        chat_assistant.documents = [test_doc]
        context = chat_assistant.get_context_from_docs("test", use_context=True)
        assert "test content" in context
class TestPromptTemplateHandling:
    @pytest.fixture(autouse=True)
    def setup(self, chat_assistant, monkeypatch, mock_config):
        self.chat_assistant = chat_assistant
        monkeypatch.setattr('ai_model_core.config.settings.load_config', 
                           lambda: mock_config)
        
    def test_prompt_template_loading(self, mock_config):
        """Test that prompt templates are loaded correctly"""
        prompt_template = get_prompt_template("code_review", mock_config)
        assert prompt_template is not None
        assert "code reviewer" in prompt_template.format(
            prompt_info="code_review",
            user_message="test"
        ).lower()

    @pytest.mark.asyncio
    async def test_chat_with_prompt_system_message(self):
        """Test that system message is included when using prompts"""
        message = "Review this code"
        prompt_info = "code_review"
        language_choice = "english"
        
        # Mock _format_history to capture the messages
        formatted_messages = []
        def mock_format_messages(*args, **kwargs):
            nonlocal formatted_messages
            formatted_messages = args[0]
            return []

        with patch('ai_model_core.shared_utils.utils._format_history', 
                  side_effect=mock_format_messages):
            async for _ in self.chat_assistant.chat(
                message=message,
                history=[],
                history_flag=True,
                stream=True,
                prompt_info=prompt_info,
                language_choice=language_choice
            ):
                pass

        # Verify system message
        assert any(isinstance(msg, SystemMessage) for msg in formatted_messages)
        system_messages = [msg for msg in formatted_messages 
                         if isinstance(msg, SystemMessage)]
        assert len(system_messages) == 1
        assert system_messages[0].content == "You are a helpful AI assistant."

    @pytest.mark.asyncio
    async def test_prompt_template_formatting(self):
        """Test that prompt templates format messages correctly"""
        message = "Check this code"
        prompt_info = "code_review"
        language_choice = "english"
        
        formatted_message = None
        def mock_format_messages(*args, **kwargs):
            nonlocal formatted_message
            messages = args[0]
            formatted_message = [msg for msg in messages 
                               if isinstance(msg, HumanMessage)][-1]
            return []

        with patch('ai_model_core.shared_utils.utils._format_history', 
                  side_effect=mock_format_messages):
            async for _ in self.chat_assistant.chat(
                message=message,
                history=[],
                history_flag=True,
                stream=True,
                prompt_info=prompt_info,
                language_choice=language_choice
            ):
                pass

        assert formatted_message is not None
        assert "code reviewer" in formatted_message.content.lower()
        assert message in formatted_message.content

    @pytest.mark.asyncio
    async def test_prompt_template_language_switching(self):
        """Test that prompt templates work with different languages"""
        message = "Check this code"
        test_cases = [
            ("english", "code_review", "code reviewer"),
            ("dutch", "analyse", "data analist")
        ]
        
        for language, prompt_info, expected_text in test_cases:
            formatted_message = None
            def mock_format_messages(*args, **kwargs):
                nonlocal formatted_message
                messages = args[0]
                formatted_message = [msg for msg in messages 
                                   if isinstance(msg, HumanMessage)][-1]
                return []

            with patch('ai_model_core.shared_utils.utils._format_history', 
                      side_effect=mock_format_messages):
                async for _ in self.chat_assistant.chat(
                    message=message,
                    history=[],
                    history_flag=True,
                    stream=True,
                    prompt_info=prompt_info,
                    language_choice=language
                ):
                    pass

            assert formatted_message is not None
            assert expected_text in formatted_message.content.lower()