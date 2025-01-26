# tests/chat_assistant/test_chat_assistant.py
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import asyncio
from typing import List, Tuple
import tempfile
import os
import shutil
from PIL import Image
import io

from langchain.schema import HumanMessage, AIMessage, Document, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr

from ai_model_core.model_helpers.chat_assistant import ChatAssistant
from ai_model_core.shared_utils.utils import ( 
    EnhancedContentLoader,
    get_prompt_template,
    get_prompt_list, 
    update_prompt_list,
    _format_history,
    format_user_message,
    format_assistant_message,
    format_file_content,
    convert_history_to_messages,
    process_message
)
from ai_model_core.config.settings import (
    load_config
)

# tests/chat_assistant/test_chat_assistant.py
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain.schema import HumanMessage, AIMessage, SystemMessage, Document
import gradio as gr

class TestGradioMessageFormat:
    def test_basic_message_structure(self):
        message = {"role": "user", "content": "Hello"}
        assert "role" in message
        assert "content" in message
        assert message["role"] in ["user", "assistant"]
        assert isinstance(message["content"], str)

    @pytest.mark.parametrize("invalid_message", [
        {"content": "Missing role"},
        {"role": "invalid", "content": "Wrong role"},
        {"role": "user"},  # Missing content
        "Not a dict"
    ])
    def test_invalid_message_formats(self, invalid_message):
        with pytest.raises((KeyError, ValueError, AssertionError)):
            self.validate_message(invalid_message)

    def test_multimodal_message_format(self):
        multimodal_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image", "path": "test.jpg"}
            ]
        }
        assert isinstance(multimodal_msg["content"], list)
        assert all("type" in component for component in multimodal_msg["content"])

    @staticmethod
    def validate_message(message):
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert message["role"] in ["user", "assistant"]

class TestMessageFormatting:
    def test_format_user_message(self):
        message = "Hello"
        history = []
        _, formatted_history = format_user_message(message, None, history)
        assert len(formatted_history) == 1
        assert formatted_history[0]["role"] == "user"
        assert formatted_history[0]["content"] == message

    def test_format_assistant_message(self):
        content = "Test response"
        metadata = {"model": "test-model"}
        message = format_assistant_message(content, metadata)
        assert message["role"] == "assistant"
        assert message["content"] == content
        assert message["metadata"] == metadata

    def test_convert_history_to_messages(self):
        langchain_history = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
            SystemMessage(content="System prompt")
        ]
        
        gradio_messages = convert_history_to_messages(langchain_history)
        assert len(gradio_messages) == 2  # System message filtered out
        assert gradio_messages[0]["role"] == "user"
        assert gradio_messages[1]["role"] == "assistant"

class TestChatAssistant:
    @pytest.mark.asyncio
    async def test_chat_with_context(self, chat_assistant):
        message = {"role": "user", "content": "Hi"}
        history = []
        
        async for response in chat_assistant.chat(
            message=message,
            history=history,
            stream=True
        ):
            assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_multimodal_chat(self, chat_assistant):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this"},
                {"type": "image", "path": "test.jpg"}
            ]
        }
        
        async for response in chat_assistant.chat(
            message=message,
            history=[],
            stream=True
        ):
            assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_file_handling(self, chat_assistant):
        test_file = Mock(spec=gr.File)
        test_file.name = "test.txt"
        
        status, success = await chat_assistant.process_chat_context_files([test_file])
        assert success is True
class TestMessageFormatValidation:
    def test_langchain_to_gradio_conversion(self):
        langchain_history = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello")
        ]
        
        gradio_format = convert_history_to_messages(langchain_history)
        assert isinstance(gradio_format, list)
        assert all({"role", "content"} <= msg.keys() for msg in gradio_format)
        assert gradio_format[0]["role"] == "user"

    def test_chat_format_integration(self, chat_assistant):
        gradio_message = {"role": "user", "content": "Hi"}
        history = []
        response = chat_assistant.chat(gradio_message, history)
        assert isinstance(response.content, str)
        
class TestMessageFormatting:
    def test_format_utils(self):
        # Test user message formatting
        message = "Hello"
        history = []
        _, formatted_history = format_user_message(message, history)
        assert formatted_history[0] == {"role": "user", "content": "Hello"}

        # Test assistant message formatting
        assistant_msg = format_assistant_message("Hi")
        assert assistant_msg == {"role": "assistant", "content": "Hi"}

        # Test conversion chain
        gradio_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        langchain_messages = _format_history(gradio_history)
        assert isinstance(langchain_messages[0], HumanMessage)
        assert isinstance(langchain_messages[1], AIMessage)
        

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

@pytest.fixture
def mock_file_processor():
    # Using Mock to simulate EnhancedContentLoader
    processor = Mock(spec=EnhancedContentLoader)
    processor.process_image.return_value = ('test_content', {'width': 100, 'height': 100})
    processor.process_document.return_value = Document(page_content='test content')
    processor.cleanup.return_value = None
    return processor

@pytest.fixture
def sample_image():
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
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
        monkeypatch.setattr('ai_model_core.config.settings.load_config', 
                           lambda: mock_config)
        return chat_assistant
        
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
class TestMessageFormatting:
    def test_format_user_message_text_only(self):
        message = "Hello"
        result, history = format_user_message(message)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == message

    def test_format_user_message_with_files(self, mock_gradio_files):
        message = "Check these files"
        result, history = format_user_message(message, mock_gradio_files)
        assert len(history) == 3  # Message + 2 files
        assert all(msg["role"] == "user" for msg in history)
        file_messages = [msg for msg in history if isinstance(msg["content"], dict)]
        assert len(file_messages) == 2
        for msg in file_messages:
            assert "path" in msg["content"]
            assert "alt_text" in msg["content"]

    def test_format_assistant_message(self):
        content = "Response"
        metadata = {"model": "test"}
        message = format_assistant_message(content, metadata)
        assert message["role"] == "assistant"
        assert message["content"] == content
        assert message["metadata"] == metadata

    def test_convert_history_formats(self):
        tuple_history = [("user msg", "assistant msg")]
        dict_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        
        tuple_messages = convert_history_to_messages(tuple_history)
        dict_messages = convert_history_to_messages(dict_history)
        
        assert len(tuple_messages) == 2
        assert len(dict_messages) == 2
        assert all("role" in msg and "content" in msg 
                  for msg in tuple_messages + dict_messages)

class TestMultimodalProcessing:
    @pytest.mark.asyncio
    async def test_image_processing(self, chat_assistant, mock_file_processor, sample_image):
        chat_assistant.content_loader = mock_file_processor
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(sample_image)
            
        try:
            response = await chat_assistant.chat(
                message={
                    'content': [
                        {'type': 'text', 'text': 'Describe this image'},
                        {'type': 'image', 'path': tmp_file.name}
                    ]
                },
                history=[]
            )
            
            mock_file_processor.process_image.assert_called_once()
            assert mock_file_processor.process_image.call_args[0][0] == tmp_file.name
            
        finally:
            os.unlink(tmp_file.name)

    @pytest.mark.asyncio
    async def test_multimodal_message(self, chat_assistant, mock_file_processor):
        chat_assistant.content_loader = mock_file_processor
        
        response = await chat_assistant.chat(
            message={
                'content': [
                    {'type': 'text', 'text': 'Look at these files'},
                    {'type': 'file', 'path': 'test.pdf'},
                    {'type': 'image', 'path': 'test.jpg'}
                ]
            },
            history=[]
        )
        
        assert mock_file_processor.process_document.called
        assert mock_file_processor.process_image.called

    @pytest.mark.asyncio
    async def test_file_size_limits(self, chat_assistant):
        large_file = gr.File(
            name='large.pdf',
            size=11*1024*1024  # 11MB
        )
        
        with pytest.raises(ValueError, match='File too large'):
            await chat_assistant.process_chat_context_files([large_file])

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, chat_assistant):
        invalid_file = gr.File(
            name='test.xyz'
        )
        
        with pytest.raises(ValueError, match='Unsupported file type'):
            await chat_assistant.process_chat_context_files([invalid_file])

class TestPromptTemplates:
    @pytest.mark.asyncio
    async def test_template_validation(self, chat_assistant):
        with pytest.raises(ValueError, match='Invalid template'):
            await chat_assistant.chat(
                message="test",
                history=[],
                prompt_info="nonexistent_template",
                language_choice="english"
            )

    @pytest.mark.asyncio
    async def test_template_interpolation(self, chat_assistant):
        test_cases = [
            ("", "empty message"),
            ("x" * 1000, "long message"),
            ("Hello\n\nWorld", "multiline message"),
            ("Special chars: !@#$%", "special characters")
        ]
        
        for message, case in test_cases:
            try:
                await chat_assistant.chat(
                    message=message,
                    history=[],
                    prompt_info="general",
                    language_choice="english"
                )
            except Exception as e:
                pytest.fail(f"Template interpolation failed for {case}: {str(e)}")

    @pytest.mark.asyncio
    async def test_language_switching_edge_cases(self, chat_assistant):
        test_cases = [
            ("english", "general"),
            ("dutch", "algemeen"),
            ("english", "code_review"),
            ("dutch", "code_review")
        ]
        
        for lang, template in test_cases:
            response = None
            async for chunk in chat_assistant.chat(
                message="test message",
                history=[],
                prompt_info=template,
                language_choice=lang
            ):
                response = chunk
            assert response is not None

    def test_template_cleanup(self, chat_assistant):
        template = get_prompt_template("general", chat_assistant.config)
        formatted = template.format(
            prompt_info="general",
            user_message="test"
        )
        assert "{" not in formatted
        assert "}" not in formatted

class TestIntegration:
    @pytest.mark.asyncio
    async def test_context_retrieval_with_files(self, chat_assistant, mock_file_processor):
        chat_assistant.content_loader = mock_file_processor
        
        # Add test documents
        chat_assistant.documents = [
            Document(page_content="test content A", metadata={"source": "doc1.pdf"}),
            Document(page_content="test content B", metadata={"source": "doc2.txt"}),
            Document(page_content="unrelated content", metadata={"source": "doc3.txt"})
        ]
        
        context = chat_assistant.get_context_from_docs(
            message="Find content A and B",
            use_context=True
        )
        
        assert "test content A" in context
        assert "test content B" in context
        assert "unrelated content" not in context

    @pytest.mark.asyncio
    async def test_streaming_with_large_response(self, chat_assistant):
        long_response = "Long response " * 100
        chat_assistant.model = MockStreamingModel(long_response)
        
        chunks = []
        async for chunk in chat_assistant.chat(
            message="Generate long response",
            history=[],
            stream=True
        ):
            chunks.append(chunk)
            
        assert len(chunks) > 0
        assert ''.join(chunks) == long_response

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self, chat_assistant, mock_file_processor):
        chat_assistant.content_loader = mock_file_processor
        temp_dir = tempfile.mkdtemp()
        chat_assistant.content_loader.temp_dir = temp_dir
        
        try:
            await chat_assistant.process_chat_context_files([
                gr.File(name='test.txt')
            ])
            mock_file_processor.cleanup.assert_called_once()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)