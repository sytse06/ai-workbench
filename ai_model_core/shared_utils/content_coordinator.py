# ai_model_core/shared_utils/content_coordinator.py
from typing import(
    List, 
    Dict, 
    Optional, 
    Union, 
    Any, 
    Tuple, 
    Callable, 
    AsyncGenerator
)
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import os

# Third party libraries
from langchain.schema import Document

# Local imports
from .content_loader import EnhancedContentLoader
from .content_processor import (
    AssistantType,
    LoaderConfig,
    BaseContentProcessor,
    ChatContentProcessor,
    RAGContentProcessor,
    SummarizationContentProcessor,
    TranscriptionContentProcessor
)

from .message_types import (
    GradioMessage,
    GradioContent,
    GradioFileContent,
    GradioRole
)
from .message_processing import MessageProcessor

logger = logging.getLogger(__name__)

class ContentProcessingComponent:
    """
    Coordinator component that manages content loading and processing
    across different assistant types.
    
    This component:
    1. Maintains configurations for different assistant types
    2. Orchestrates the loading and processing flow
    3. Routes content to appropriate processors
    4. Manages callbacks for assistant-specific operations
    
    Implements the Singleton pattern to ensure a single instance
    throughout the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContentProcessingComponent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ContentProcessingComponent."""
        if self._initialized:
            return
            
        # Configuration for different assistant types
        self._configs = {}
        
        # Content loader instance
        self._content_loader = None
        
        # Processor instances for different assistant types
        self._processors = {}
        
        # State flag
        self._initialized = True
        
    def initialize_loader(
        self,
        temp_dir: str = "input/tmp",
        audio_sample_rate: int = 16000,
        perform_ocr: bool = True
    ):
        """
        Initialize the underlying EnhancedContentLoader.
        
        Args:
            temp_dir: Directory for temporary files
            audio_sample_rate: Sample rate for audio processing
            perform_ocr: Whether to perform OCR on images
        """
        self._content_loader = EnhancedContentLoader(
            temp_dir=temp_dir,
            audio_sample_rate=audio_sample_rate,
            perform_ocr=perform_ocr
        )
        
        # Initialize default processors
        self._initialize_default_processors()
    
    def _initialize_default_processors(self):
        """Initialize default processors for each assistant type."""
        # Only initialize if content loader exists
        if not self._content_loader:
            return
            
        # Create processor instances
        self._processors[AssistantType.CHAT] = ChatContentProcessor(self._content_loader)
        self._processors[AssistantType.RAG] = RAGContentProcessor(self._content_loader)
        self._processors[AssistantType.SUMMARIZATION] = SummarizationContentProcessor(self._content_loader)
        self._processors[AssistantType.TRANSCRIPTION] = TranscriptionContentProcessor(self._content_loader)
    
    def register_assistant(
        self,
        assistant_type: AssistantType,
        config: LoaderConfig
    ):
        """
        Register an assistant with its specific configuration.
        
        Args:
            assistant_type: Type of assistant
            config: Configuration for loading and processing
        """
        try:
            self._configs[assistant_type] = config
            
            # Create content loader if not already initialized
            if self._content_loader is None:
                self.initialize_loader()
            
        except Exception as e:
            logger.error(f"Error registering assistant {assistant_type}: {str(e)}")
            raise
    
    def register_processor(
        self,
        assistant_type: AssistantType,
        processor: BaseContentProcessor
    ):
        """
        Register a custom processor for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            processor: Custom content processor
        """
        self._processors[assistant_type] = processor
    
    async def process_content(
        self,
        assistant_type: AssistantType,
        url_input: Optional[str] = None,
        file_input: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Optional[Any]]:
        """
        Process content through loading and assistant-specific processing.
        
        Args:
            assistant_type: Type of assistant
            url_input: Optional URL string
            file_input: Optional file path(s)
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (status message, processed result)
        """
        try:
            # Validate assistant type
            if assistant_type not in self._configs:
                return f"Assistant type {assistant_type} not registered.", None
            
            # Get config for this assistant type
            config = self._configs[assistant_type]
            
            # Initialize loader if needed
            if not self._content_loader:
                self.initialize_loader()
            
            # Use provided chunk parameters or fall back to config
            effective_chunk_size = chunk_size or config.chunk_size
            effective_chunk_overlap = chunk_overlap or config.chunk_overlap
            
            # Load documents first
            try:
                # For audio files, allow potentially different loading behavior
                if assistant_type == AssistantType.TRANSCRIPTION:
                    return await self._process_transcription_content(
                        url_input, file_input, config, **kwargs)
                
                # Standard document loading and chunking
                docs = await asyncio.to_thread(
                    self._content_loader.load_and_chunk_documents,
                    file_paths=file_input,
                    urls=url_input,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=effective_chunk_overlap
                )
                
                if not docs:
                    return "No documents were loaded", None
                
                # Process through appropriate processor
                processor = self._processors.get(assistant_type)
                if not processor:
                    logger.warning(f"No processor registered for {assistant_type}")
                    
                    # Call callback directly if no processor but callback exists
                    if config.process_callback:
                        result = await self._execute_callback(config.process_callback, docs)
                        return f"Processed {len(docs)} documents (callback only)", result
                    
                    return f"No processor for {assistant_type}", docs
                
                # Process through the appropriate processor
                status, result = await processor.process_documents(docs, **kwargs)
                
                # Execute callback if provided
                if config.process_callback:
                    await self._execute_callback(config.process_callback, docs)
                
                return status, result
                
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                return f"Error processing documents: {str(e)}", None
                
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}")
            return f"Error in content processing: {str(e)}", None

    async def process_content_for_messages(
        self,
        assistant_type: AssistantType,
        file_input: Optional[Union[str, List[str]]] = None,
        url_input: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> Tuple[GradioMessage, Optional[GradioMessage]]:
        """
        Process content and return formatted messages for chat interface.
        
        Args:
            assistant_type: Type of assistant
            file_input: Optional file path(s)
            url_input: Optional URL
            query: Optional query for relevance-based formatting
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (system message with context, file notification message)
        """
        try:
            status, result = await self.process_content(
                assistant_type=assistant_type,
                file_input=file_input,
                url_input=url_input,
                query=query,
                **kwargs
            )
            
            # Extract context if available
            context = ""
            files_message = None
            
            if isinstance(result, dict):
                context = result.get("context", "")
                
                # Create file notification message if available
                if "files_message" in result and result["files_message"]:
                    files_message = self._message_processor.format_system_message(
                        result["files_message"]
                    )
            elif result:
                context = str(result)
                
            # If no context, use status message
            if not context:
                context = status
                
            # Create system message with context
            system_message = self._message_processor.format_system_message(
                f"Context:\n{context}" if context else status
            )
            
            return system_message, files_message
            
        except Exception as e:
            logger.error(f"Error creating context message: {str(e)}")
            error_message = self._message_processor.format_system_message(
                f"Error processing content: {str(e)}"
            )
            return error_message, None
            
    async def create_file_notification_message(
        self,
        assistant_type: AssistantType,
        files: List[str]
    ) -> Optional[GradioMessage]:
        """
        Create a notification message about processed files.
        
        Args:
            assistant_type: Type of assistant
            files: List of file paths that were processed
            
        Returns:
            GradioMessage with file notification or None if no files
        """
        if not files:
            return None
            
        try:
            # Get processor for this assistant type
            processor = self._processors.get(assistant_type)
            if not processor:
                return None
                
            # Process the files to get metadata (without callbacks)
            docs = await asyncio.to_thread(
                self._content_loader.load_documents,
                file_paths=files
            )
            
            if not docs:
                return None
                
            # Generate file message
            files_message = processor.format_files_message(docs)
            
            # Create system message
            return self._message_processor.format_system_message(files_message)
            
        except Exception as e:
            logger.error(f"Error creating file notification: {str(e)}")
            return None
    
    async def _process_transcription_content(
        self,
        url_input: Optional[str],
        file_input: Optional[Union[str, List[str]]],
        config: LoaderConfig,
        **kwargs
    ) -> Tuple[str, Optional[Any]]:
        """
        Special handling for transcription content.
        
        Args:
            url_input: Optional URL string
            file_input: Optional file path(s)
            config: Loader configuration
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (status message, processed result)
        """
        try:
            # Directly load audio files without chunking
            docs = await asyncio.to_thread(
                self._content_loader.load_documents,
                file_paths=file_input,
                urls=url_input
            )
            
            if not docs:
                return "No audio files were loaded", None
            
            # Process through transcription processor
            processor = self._processors.get(AssistantType.TRANSCRIPTION)
            if not processor:
                # Call callback directly if no processor but callback exists
                if config.process_callback:
                    result = await self._execute_callback(config.process_callback, docs)
                    return f"Processed {len(docs)} audio files (callback only)", result
                
                return "No transcription processor registered", docs
            
            # Process through the transcription processor
            status, result = await processor.process_documents(docs, **kwargs)
            
            # Execute callback if provided
            if config.process_callback:
                await self._execute_callback(config.process_callback, docs)
            
            return status, result
            
        except Exception as e:
            logger.error(f"Error processing transcription content: {str(e)}")
            return f"Error processing transcription content: {str(e)}", None
    
    async def _execute_callback(self, callback: Callable, docs: List[Document]) -> Any:
        """
        Execute a callback function with document list.
        
        Args:
            callback: Function to call with documents
            docs: List of Document objects
            
        Returns:
            Result from the callback
        """
        try:
            # Handle both sync and async callbacks
            if asyncio.iscoroutinefunction(callback):
                # Async callback
                return await callback(docs)
            else:
                # Sync callback
                return await asyncio.to_thread(callback, docs)
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
            raise
    
    def get_loader_config(self, assistant_type: AssistantType) -> Optional[LoaderConfig]:
        """
        Get the current configuration for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            
        Returns:
            LoaderConfig for the assistant type, or None if not registered
        """
        return self._configs.get(assistant_type)
    
    @property
    def content_loader(self) -> EnhancedContentLoader:
        """
        Access the underlying EnhancedContentLoader.
        
        Returns:
            The EnhancedContentLoader instance
        """
        if self._content_loader is None:
            self.initialize_loader()
        return self._content_loader
    
    def get_processor(self, assistant_type: AssistantType) -> Optional[BaseContentProcessor]:
        """
        Get the processor for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            
        Returns:
            BaseContentProcessor for the assistant type, or None if not registered
        """
        return self._processors.get(assistant_type)
    
    def get_message_processor(self) -> MessageProcessor:
        """
        Get the message processor instance.
        
        Returns:
            The MessageProcessor instance
        """
        return self._message_processor
    
    def register_processor(
        self,
        assistant_type: AssistantType,
        processor: BaseContentProcessor
    ):
        """
        Register a custom processor for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            processor: Custom content processor
        """
        self._processors[assistant_type] = processor
    
    async def process_content(
        self,
        assistant_type: AssistantType,
        url_input: Optional[str] = None,
        file_input: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Optional[Any]]:
        """
        Process content through loading and assistant-specific processing.
        
        Args:
            assistant_type: Type of assistant
            url_input: Optional URL string
            file_input: Optional file path(s)
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (status message, processed result)
        """
        try:
            # Validate assistant type
            if assistant_type not in self._configs:
                return f"Assistant type {assistant_type} not registered.", None
            
            # Get config for this assistant type
            config = self._configs[assistant_type]
            
            # Initialize loader if needed
            if not self._content_loader:
                self.initialize_loader()
            
            # Use provided chunk parameters or fall back to config
            effective_chunk_size = chunk_size or config.chunk_size
            effective_chunk_overlap = chunk_overlap or config.chunk_overlap
            
            # Load documents first
            try:
                # For audio files, allow potentially different loading behavior
                if assistant_type == AssistantType.TRANSCRIPTION:
                    return await self._process_transcription_content(
                        url_input, file_input, config, **kwargs)
                
                # Standard document loading and chunking
                docs = await asyncio.to_thread(
                    self._content_loader.load_and_chunk_documents,
                    file_paths=file_input,
                    urls=url_input,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=effective_chunk_overlap
                )
                
                if not docs:
                    return "No documents were loaded", None
                
                # Process through appropriate processor
                processor = self._processors.get(assistant_type)
                if not processor:
                    logger.warning(f"No processor registered for {assistant_type}")
                    
                    # Call callback directly if no processor but callback exists
                    if config.process_callback:
                        result = await self._execute_callback(config.process_callback, docs)
                        return f"Processed {len(docs)} documents (callback only)", result
                    
                    return f"No processor for {assistant_type}", docs
                
                # Process through the appropriate processor
                status, result = await processor.process_documents(docs, **kwargs)
                
                # Execute callback if provided
                if config.process_callback:
                    await self._execute_callback(config.process_callback, docs)
                
                return status, result
                
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                return f"Error processing documents: {str(e)}", None
                
        except Exception as e:
            logger.error(f"Error in process_content: {str(e)}")
            return f"Error in content processing: {str(e)}", None
    
    async def _process_transcription_content(
        self,
        url_input: Optional[str],
        file_input: Optional[Union[str, List[str]]],
        config: LoaderConfig,
        **kwargs
    ) -> Tuple[str, Optional[Any]]:
        """
        Special handling for transcription content.
        
        Args:
            url_input: Optional URL string
            file_input: Optional file path(s)
            config: Loader configuration
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (status message, processed result)
        """
        try:
            # Directly load audio files without chunking
            docs = await asyncio.to_thread(
                self._content_loader.load_documents,
                file_paths=file_input,
                urls=url_input
            )
            
            if not docs:
                return "No audio files were loaded", None
            
            # Process through transcription processor
            processor = self._processors.get(AssistantType.TRANSCRIPTION)
            if not processor:
                # Call callback directly if no processor but callback exists
                if config.process_callback:
                    result = await self._execute_callback(config.process_callback, docs)
                    return f"Processed {len(docs)} audio files (callback only)", result
                
                return "No transcription processor registered", docs
            
            # Process through the transcription processor
            status, result = await processor.process_documents(docs, **kwargs)
            
            # Execute callback if provided
            if config.process_callback:
                await self._execute_callback(config.process_callback, docs)
            
            return status, result
            
        except Exception as e:
            logger.error(f"Error processing transcription content: {str(e)}")
            return f"Error processing transcription content: {str(e)}", None
    
    async def _execute_callback(self, callback: Callable, docs: List[Document]) -> Any:
        """
        Execute a callback function with document list.
        
        Args:
            callback: Function to call with documents
            docs: List of Document objects
            
        Returns:
            Result from the callback
        """
        try:
            # Handle both sync and async callbacks
            if asyncio.iscoroutinefunction(callback):
                # Async callback
                return await callback(docs)
            else:
                # Sync callback
                return await asyncio.to_thread(callback, docs)
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
            raise
    
    def get_loader_config(self, assistant_type: AssistantType) -> Optional[LoaderConfig]:
        """
        Get the current configuration for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            
        Returns:
            LoaderConfig for the assistant type, or None if not registered
        """
        return self._configs.get(assistant_type)
    
    @property
    def content_loader(self) -> EnhancedContentLoader:
        """
        Access the underlying EnhancedContentLoader.
        
        Returns:
            The EnhancedContentLoader instance
        """
        if self._content_loader is None:
            self.initialize_loader()
        return self._content_loader
    
    def get_processor(self, assistant_type: AssistantType) -> Optional[BaseContentProcessor]:
        """
        Get the processor for an assistant type.
        
        Args:
            assistant_type: Type of assistant
            
        Returns:
            BaseContentProcessor for the assistant type, or None if not registered
        """
        return self._processors.get(assistant_type)

# Helper function to set up content processing during application startup
def setup_content_processing(app_config: Dict[str, Any]) -> ContentProcessingComponent:
    """
    Initialize and configure the content processing component.
    
    Args:
        app_config: Application configuration dictionary
        
    Returns:
        Configured ContentProcessingComponent instance
    """
    processor = ContentProcessingComponent()
    
    # Configure for CHAT processing
    chat_config = LoaderConfig(
        chunk_size=app_config.get("chat", {}).get("chunk_size", 1000),
        chunk_overlap=app_config.get("chat", {}).get("chunk_overlap", 200),
        process_callback=None  # Will be set after Chat assistant initialization
    )
    processor.register_assistant(AssistantType.CHAT, chat_config)
    
    # Configure for RAG processing
    rag_config = LoaderConfig(
        chunk_size=app_config.get("rag", {}).get("chunk_size", 500),
        chunk_overlap=app_config.get("rag", {}).get("chunk_overlap", 50),
        process_callback=None  # Will be set after RAG assistant initialization
    )
    processor.register_assistant(AssistantType.RAG, rag_config)
    
    # Configure for SUMMARIZATION processing
    summary_config = LoaderConfig(
        chunk_size=app_config.get("summarization", {}).get("chunk_size", 1000),
        chunk_overlap=app_config.get("summarization", {}).get("chunk_overlap", 100),
        process_callback=None  # Will be set after Summarization assistant initialization
    )
    processor.register_assistant(AssistantType.SUMMARIZATION, summary_config)
    
    # Configure for TRANSCRIPTION processing
    transcription_config = LoaderConfig(
        chunk_size=app_config.get("transcription", {}).get("chunk_size", 30000),
        chunk_overlap=app_config.get("transcription", {}).get("chunk_overlap", 0),
        process_callback=None  # Will be set after Transcription assistant initialization
    )
    processor.register_assistant(AssistantType.TRANSCRIPTION, transcription_config)
    
    # Initialize the content loader with config settings
    processor.initialize_loader(
        temp_dir=app_config.get("directories", {}).get("temp", "input/tmp"),
        audio_sample_rate=app_config.get("audio", {}).get("sample_rate", 16000),
        perform_ocr=app_config.get("ocr", {}).get("enabled", True)
    )
    
    return processor