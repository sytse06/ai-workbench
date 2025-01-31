import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os
import shutil
from PIL import Image
import io

from langchain.schema import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    Document
)

import gradio as gr

from ai_model_core.model_helpers.chat_assistant import ChatAssistant
from ai_model_core.shared_utils.utils import (
    EnhancedContentLoader,
    get_prompt_template,
    format_user_message,
    format_assistant_message,
    convert_history_to_messages
)

class MockStreamingModel:
    """Mock model that simulates LangChain streaming interface"""
    def __init__(self, response="Mock response"):
        self.response = response
    
    def bind(self, **kwargs):
        return self
        
    async def astream(self, messages, **kwargs):
        yield AIMessage(content=self.response)
        
    async def agenerate(self, messages, **kwargs):
        return MockLLMResult(generations=[[MockGeneration(text=self.response)]])

class MockLLMResult:
    def __init__(self, generations):
        self.generations = generations

class MockGeneration:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def mock_config():
    return {
        "system_prompt_settings": {
            "english": {"system_prompt": "You are a helpful AI assistant."},
            "dutch": {"system_prompt": "Je bent een behulpzame AI-assistent."}
        },
        "prompts": {
            "english": ["general", "code_review"],
            "dutch": ["algemeen", "code_review"],
            "general": "Act as a general assistant.",
            "code_review": "Act as a code reviewer."
        }
    }

@pytest.fixture
def chat_assistant():
    with patch('ai_model_core.shared_utils.factory.get_model', 
              return_value=MockStreamingModel()):
        assistant = ChatAssistant(
            model_choice="test_model",
            temperature=0.7,
            max_tokens=500
        )
        assistant.content_loader = MagicMock(spec=EnhancedContentLoader)
        assistant.content_loader.load_documents.return_value = [
            Document(page_content="test content", metadata={"source": "test.txt"})
        ]
        return assistant

@pytest.fixture
def mock_gradio_files():
    file1 = MagicMock(spec=gr.File)
    file1.name = str(Path("test1.txt").absolute())
    file2 = MagicMock(spec=gr.File)
    file2.name = str(Path("test2.txt").absolute())
    return [file1, file2]

@pytest.mark.asyncio
class TestMessageFormatting:
    async def test_format_user_message(self):
        """Test user message formatting function."""
        message = "Hello"
        history = []
        empty_str, new_history = await format_user_message(message, None, history)
        
        assert empty_str == ""
        assert len(new_history) == 1
        assert new_history[0]["role"] == "user"
        assert new_history[0]["content"] == message
        
    async def test_format_user_message_with_files(self, chat_assistant):
        files = [MagicMock(spec=gr.File) for _ in range(2)]
        for i, f in enumerate(files):
            f.name = f"test{i}.txt"

        result = await format_user_message("Check these files", files, [])
        empty_str, history = result

        assert empty_str == ""
        assert len(history) == 3
        assert isinstance(history[-1]["content"], str)
        assert history[-1]["content"] == "Check these files"

    def test_format_assistant_message(self):
        content = "Test response"
        metadata = {"model": "test-model"}
        message = format_assistant_message(content, metadata)
        assert message["role"] == "assistant"
        assert message["content"] == content
        assert message["metadata"] == metadata

    def test_convert_history_format(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ]
        messages = convert_history_to_messages(history)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello"
        assert all("role" in msg and "content" in msg for msg in messages)

class TestChatAssistant:
    @pytest.mark.asyncio
    async def test_initialization(self, chat_assistant):
        assert chat_assistant.model_choice == "test_model"
        assert chat_assistant.temperature == 0.7
        assert chat_assistant.max_tokens == 500
        assert isinstance(chat_assistant.documents, list)
    
    @pytest.mark.asyncio
    async def test_file_validation(self, monkeypatch, chat_assistant, test_input_dir):
        class MockStat:
            def __init__(self, size):
                self.st_size = size
                self.st_mode = 0o777  # Add st_mode to simulate a directory

            def __call__(self, *, follow_symlinks=True):
                return self

        test_cases = [
            ([("image.jpg", 5 * 1024 * 1024)], True),           # Single valid image
            ([("big.jpg", 5 * 1024 * 1024 + 1)], False),        # Single oversized image
            ([("doc.txt", 10 * 1024 * 1024)], True),            # Single valid text
            ([("big.txt", 10 * 1024 * 1024 + 1)], False),       # Single oversized text
            ([("doc1.txt", 5 * 1024 * 1024),                    # Combined size test
            ("doc2.txt", 5 * 1024 * 1024 + 1)], False)
        ]

        for files_info, should_pass in test_cases:
            files = []
            for name, size in files_info:
                # Create a temporary file with the specified size
                file_path = test_input_dir / name
                with open(file_path, 'wb') as f:
                    f.write(b'\0' * size)
                
                mock_file = MagicMock(spec=gr.File)
                mock_file.name = str(file_path)
                files.append(mock_file)
                monkeypatch.setattr(Path, "stat", MockStat(size))

            if should_pass:
                await chat_assistant._validate_files(files)
            else:
                with pytest.raises(ValueError):
                    await chat_assistant._validate_files(files)
                                                                            
    @pytest.mark.asyncio
    async def test_detect_file_type(self):
        """Test file type detection logic"""
        test_cases = [
            ("doc.txt", "text"),
            ("script.py", "text"),
            ("image.jpg", "image"),
            ("photo.png", "image"),
            ("data.json", "text"),
            ("unknown.xyz", "text")  # Default to text
        ]
        
        for filename, expected_type in test_cases:
            mock_file = MagicMock(spec=gr.File)
            mock_file.name = filename
            detected_type = chat_assistant.detect_file_type(mock_file)
            assert detected_type == expected_type
        
    @pytest.mark.asyncio
    async def test_word_count_limit(self, chat_assistant, test_input_dir):
        # Create file with exactly 4000 words
        valid_file = test_input_dir / "valid_words.txt"
        valid_file.write_text(" ".join(["word"] * 4000))

        # Create file with 4001 words
        invalid_file = test_input_dir / "invalid_words.txt"
        invalid_file.write_text(" ".join(["word"] * 4001))

        # Test valid file
        mock_valid = MagicMock(spec=gr.File)
        mock_valid.name = str(valid_file)
        result = await chat_assistant.process_chat_context_files([mock_valid])
        assert result is not None

        # Test file exceeding word limit
        mock_invalid = MagicMock(spec=gr.File)
        mock_invalid.name = str(invalid_file)
        with pytest.raises(ValueError, match=r".*exceeds word limit of 4000"):
            await chat_assistant.process_chat_context_files([mock_invalid])

    @pytest.mark.asyncio
    async def test_combined_file_limits(self, chat_assistant, tmp_path):
        # Create two 6MB files (total 12MB > 10MB limit)
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_bytes(b"0" * (6 * 1024 * 1024))
        file2.write_bytes(b"0" * (6 * 1024 * 1024))

        mock_file1 = MagicMock(spec=gr.File)
        mock_file1.name = str(file1)
        mock_file2 = MagicMock(spec=gr.File)
        mock_file2.name = str(file2)

        with pytest.raises(ValueError, match=r"Combined file size exceeds limit"):
            await chat_assistant.process_chat_context_files([mock_file1, mock_file2])

    @pytest.mark.asyncio
    async def test_update_model(self, chat_assistant):
        with patch('ai_model_core.shared_utils.factory.get_model', 
                  return_value=MockStreamingModel()):
            new_model = "new_test_model"
            await chat_assistant.update_model(new_model)
            assert chat_assistant.model_choice == new_model

    @pytest.mark.asyncio
    async def test_chat_with_context(self, chat_assistant):
        message = {"role": "user", "content": "Test message"}
        history = []
        responses = []
        
        async for response in chat_assistant.chat(
            message=message,
            history=history,
            stream=True,
            use_context=True
        ):
            responses.append(response)
            
        assert len(responses) > 0
        assert isinstance(responses[0], str)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("message,history,history_flag", [
        ("Hello", [], True),
        ("Hi", [{"role": "user", "content": "Hello"}, 
                {"role": "assistant", "content": "Hi!"}], True),
        ("Test", [], False)
    ])
    async def test_chat_history_handling(self, chat_assistant, message, history, history_flag):
        chat_assistant.model = MockStreamingModel("Test response")
        
        responses = []
        async for response in chat_assistant.chat(
            message=message,
            history=history,
            history_flag=history_flag,
            stream=True
        ):
            responses.append(response)
            
        assert len(responses) > 0
        assert isinstance(responses[0], str)

    def test_get_context_from_docs(self, chat_assistant):
        # Test with no context
        assert chat_assistant.get_context_from_docs("test", use_context=False) == ""
        
        # Test with context
        test_docs = [
            Document(page_content="relevant test content", metadata={"source": "doc1.txt"}),
            Document(page_content="unrelated content", metadata={"source": "doc2.txt"})
        ]
        chat_assistant.documents = test_docs
        
        context = chat_assistant.get_context_from_docs("test", use_context=True)
        assert "relevant test content" in context
        assert "unrelated content" not in context

class TestPromptTemplateHandling:
    @pytest.mark.asyncio
    async def test_chat_with_prompt_template(self, chat_assistant, mock_config):
        with patch('ai_model_core.config.settings.load_config', 
                  return_value=mock_config):
            message = "Review this code"
            responses = []
            
            async for response in chat_assistant.chat(
                message=message,
                history=[],
                prompt_info="code_review",
                language_choice="english",
                stream=True
            ):
                responses.append(response)
                
            assert len(responses) > 0
            assert isinstance(responses[0], str)

    @pytest.mark.asyncio
    async def test_language_switching(self, chat_assistant, mock_config):
        with patch('ai_model_core.config.settings.load_config', 
                  return_value=mock_config):
            test_cases = [
                ("english", "code_review"),
                ("dutch", "code_review")
            ]
            
            for language, prompt_info in test_cases:
                responses = []
                async for response in chat_assistant.chat(
                    message="Test message",
                    history=[],
                    prompt_info=prompt_info,
                    language_choice=language,
                    stream=True
                ):
                    responses.append(response)
                    
                assert len(responses) > 0
                assert isinstance(responses[0], str)

class TestIntegration:
    @pytest.mark.asyncio
    async def test_streaming_with_large_response(self, chat_assistant):
        test_response = "Long response " * 100
        chat_assistant.model = MockStreamingModel(test_response)
        
        chunks = []
        async for chunk in chat_assistant.chat(
            message={"role": "user", "content": "Generate long response"},
            history=[],
            stream=True
        ):
            chunks.append(chunk)
            
        complete_response = "".join(chunks)
        assert len(complete_response) > 0
        assert complete_response == test_response

    @pytest.mark.asyncio
    async def test_context_retrieval_with_files(self, chat_assistant, test_input_dir):
        # Create test files
        test_file1 = test_input_dir / "test1.txt"
        test_file2 = test_input_dir / "test2.txt"
        
        test_file1.write_text("test content A")
        test_file2.write_text("test content B")
        
        # Create mock files
        mock_files = [
            MagicMock(spec=gr.File),
            MagicMock(spec=gr.File)
        ]
        mock_files[0].name = str(test_file1)
        mock_files[1].name = str(test_file2)
        
        # Setup chat assistant documents
        chat_assistant.documents = [
            Document(page_content="test content A", metadata={"source": str(test_file1)}),
            Document(page_content="test content B", metadata={"source": str(test_file2)})
        ]
        
        # Process files and verify context
        await chat_assistant.process_chat_context_files(mock_files)
        context = chat_assistant.get_context_from_docs(
            message="Find content A and B",
            use_context=True
        )
        
        assert "test content A" in context
        assert "test content B" in context

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self, chat_assistant, test_input_dir):
        # Create test file
        test_file = test_input_dir / "temp_test.txt"
        test_file.write_text("temporary test content")
        
        mock_file = MagicMock(spec=gr.File)
        mock_file.name = str(test_file)
        
        # Process file
        await chat_assistant.process_chat_context_files([mock_file])
        
        # Verify cleanup
        chat_assistant.content_loader.cleanup()
        assert chat_assistant.content_loader.cleanup.called
        
        # Verify directory is clean except for .gitkeep
        remaining_files = [f for f in test_input_dir.glob("*") if f.name != ".gitkeep"]
        assert len(remaining_files) == 1  # Only our test file