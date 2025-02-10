# test_shared_utils/test_enhanced_content_loader.py

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from PIL import Image
import io

from ai_model_core.shared_utils.utils import EnhancedContentLoader

class TestEnhancedContentLoader:
    """Test suite for EnhancedContentLoader functionality."""
    
    @pytest.fixture
    def loader(self, test_input_dir):
        """Create a loader instance for testing."""
        return EnhancedContentLoader(
            chunk_size=1000,
            chunk_overlap=200,
            temp_dir=str(test_input_dir)
        )

    def test_supported_formats(self, loader):
        """Test supported file format detection."""
        assert ".txt" in loader.supported_text_formats
        assert ".jpg" in loader.supported_image_formats
        assert ".mp3" in loader.supported_audio_formats
        
        # Test format validation
        assert loader.is_valid_file_type("test.txt")
        assert loader.is_valid_file_type("test.jpg")
        assert not loader.is_valid_file_type("test.invalid")

    def test_text_file_loading(self, loader, test_input_dir):
        """Test loading text files."""
        # Create test file
        test_file = test_input_dir / "test.txt"
        content = "Test content\nMultiple lines\nMore content"
        test_file.write_text(content)
        
        # Load document
        docs = loader.load_documents([str(test_file)])
        
        assert len(docs) == 1
        assert docs[0].page_content == content
        assert docs[0].metadata["source"] == str(test_file)

    def test_image_processing(self, loader, test_input_dir):
        """Test image file processing."""
        # Create test image
        test_image = test_input_dir / "test.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(test_image)
        
        # Process image
        docs = loader._process_image_file(str(test_image))
        
        assert len(docs) >= 1
        assert "Image file:" in docs[0].page_content
        assert docs[0].metadata["type"] == "image"

    def test_file_cleanup(self, loader, test_input_dir):
        """Test temporary file cleanup."""
        # Create test files
        test_files = [
            test_input_dir / "test1.txt",
            test_input_dir / "test2.txt"
        ]
        for file in test_files:
            file.write_text("test content")
            
        # Process files
        loader.load_documents([str(f) for f in test_files])
        
        # Cleanup
        loader.cleanup()
        
        # Check cleanup
        remaining_files = list(test_input_dir.glob("*"))
        assert len(remaining_files) == 0

    def test_context_processing(self, loader):
        """Test processing documents into context."""
        # Create test documents
        docs = [
            Document(
                page_content="Text content",
                metadata={
                    "source": "test.txt",
                    "type": "text"
                }
            ),
            Document(
                page_content="OCR text",
                metadata={
                    "source": "test.jpg",
                    "type": "image",
                    "has_ocr": True,
                    "file_name": "test.jpg"
                }
            )
        ]
        
        # Process context
        context = loader.process_files_for_context(docs)
        
        assert "text_content" in context
        assert "image_descriptions" in context
        assert "Text content" in context["text_content"][0]
        assert "OCR text" in context["image_descriptions"][0]

# test_model_helpers/test_chat_assistant.py
# Add these tests to the existing TestChatAssistant class

class TestChatAssistantFileHandling:
    """Tests for ChatAssistant file handling capabilities."""
    
    @pytest.mark.asyncio
    async def test_chat_with_file_context(self, chat_assistant, test_input_dir):
        """Test chat with file context."""
        # Create test file
        test_file = test_input_dir / "context.txt"
        test_file.write_text("Important information about topics")
        
        # Create mock gradio file
        mock_file = MagicMock()
        mock_file.name = str(test_file)
        
        # Process context files
        docs = await chat_assistant.process_chat_context_files([mock_file])
        assert docs
        
        # Test chat with context
        message = GradioMessage(
            role="user",
            content="What topics are mentioned?"
        )
        
        responses = []
        async for response in chat_assistant.chat(
            message=message,
            history=[],
            use_context=True
        ):
            responses.append(response)
            
        assert responses
        assert isinstance(responses[0], (str, GradioMessage))

    @pytest.mark.asyncio
    async def test_multimodal_chat(self, chat_assistant, test_input_dir):
        """Test chat with multiple file types."""
        # Create test files
        text_file = test_input_dir / "test.txt"
        text_file.write_text("Text content")
        
        image_file = test_input_dir / "test.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(image_file)
        
        # Create mock gradio files
        mock_files = [
            MagicMock(name=str(text_file)),
            MagicMock(name=str(image_file))
        ]
        
        # Process files
        docs = await chat_assistant.process_chat_context_files(mock_files)
        assert len(docs) >= 2
        
        # Test chat
        message = GradioMessage(
            role="user",
            content=[
                "Analyze these files",
                {"type": "file", "path": str(text_file)},
                {"type": "image", "path": str(image_file)}
            ]
        )
        
        async for response in chat_assistant.chat(
            message=message,
            history=[],
            use_context=True
        ):
            assert isinstance(response, (str, GradioMessage))

# test_shared_utils/test_message_processing.py
# Add these tests to TestMessageProcessor

class TestMessageProcessorWithFiles:
    """Tests for message processing with file content."""
    
    @pytest.mark.asyncio
    async def test_process_file_content(self, processor, test_input_dir):
        """Test processing messages with file content."""
        # Create test file
        test_file = test_input_dir / "test.txt"
        test_file.write_text("Test content")
        
        # Create message with file
        file_content = {
            "type": "file",
            "path": str(test_file)
        }
        message = GradioMessage(role="user", content=file_content)
        
        # Process content
        processed = await processor.process_message_content(message.content)
        assert "[File:" in processed
        
    @pytest.mark.asyncio
    async def test_process_multimodal_content(self, processor, test_input_dir):
        """Test processing multimodal messages."""
        # Create test files
        text_file = test_input_dir / "test.txt"
        text_file.write_text("Text content")
        
        # Create multimodal message
        content = [
            "Check this file",
            {"type": "file", "path": str(text_file)}
        ]
        message = GradioMessage(role="user", content=content)
        
        # Process content
        processed = await processor.process_message_content(message.content)
        assert "Check this file" in processed
        assert "[File:" in processed