# ai_model_core/shared_utils/content_loader.py
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os
import logging
import mimetypes
import tempfile
from urllib.parse import urlparse, parse_qs
from enum import Enum
import asyncio

import requests
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydub import AudioSegment
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import yt_dlp

logger = logging.getLogger(__name__)

class EnhancedContentLoader:
    """
    A focused content loader responsible for loading various file types
    and converting them to Document objects.
    
    This class handles raw file loading operations without assistant-specific processing.
    """
    # File upload constraints
    MAX_TEXT_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    MAX_IMAGE_FILE_SIZE = 5 * 1024 * 1024   # 5MB in bytes
    MAX_WORD_COUNT = 4000
    MAX_COMBINED_SIZE = 10 * 1024 * 1024    # 10MB total
    def __init__(
        self,
        temp_dir: str = "input/tmp",
        audio_sample_rate: int = 16000,
        perform_ocr: bool = True
    ):
        """
        Initialize the content loader.
        
        Args:
            temp_dir: Directory for temporary files
            audio_sample_rate: Sample rate for audio processing
            perform_ocr: Whether to perform OCR on images
        """
        self.temp_dir = Path(temp_dir)
        self.audio_sample_rate = audio_sample_rate
        self.perform_ocr = perform_ocr
        
        # Supported formats
        self.supported_audio_formats = {'.mp4', '.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        self.supported_text_formats = {'.txt', '.pdf', '.docx', '.md', '.py'}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif'}
        
        # Create temporary directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the class."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # ----------------------
    # Core Loading Functions
    # ----------------------
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in self.supported_text_formats:
                return self._load_text_document(file_path, file_extension)
            elif file_extension in self.supported_image_formats:
                return self._load_image_document(file_path)
            elif file_extension in self.supported_audio_formats:
                return self._load_audio_document(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def load_documents(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None
    ) -> List[Document]:
        """
        Load multiple documents from files and/or URLs.
        
        Args:
            file_paths: Single file path or list of file paths
            urls: String containing URLs (one per line)
            
        Returns:
            List of Document objects
        """
        docs = []
        
        try:
            # Process URLs if provided
            if urls:
                url_docs = self._load_from_urls(urls)
                if url_docs:
                    docs.extend(url_docs)

            # Process files if provided
            if file_paths:
                file_docs = self._load_from_files(file_paths)
                if file_docs:
                    docs.extend(file_docs)

            if not docs:
                raise ValueError("No documents were successfully loaded.")

            return docs

        except Exception as e:
            logger.error(f"Error in load_documents: {str(e)}")
            raise
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of chunked Document objects
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def load_and_chunk_documents(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Load documents and split them into chunks in one operation.
        
        Args:
            file_paths: Single file path or list of file paths
            urls: String containing URLs (one per line)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of chunked Document objects
        """
        try:
            docs = self.load_documents(file_paths, urls)
            return self.chunk_documents(docs, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"Error in load_and_chunk_documents: {str(e)}")
            raise
    
    # ----------------------
    # Validation Functions
    # ----------------------
    
    async def validate_files(self, files: List[Any]) -> None:
        """Validate all files against size and content limits."""
        combined_size = 0
        
        for file in files:
            file_path = file.name if hasattr(file, "name") else str(file)
            file_size = os.path.getsize(file_path)
            combined_size += file_size
            
            # Check individual file size limits
            if self._is_image_file(file_path):
                if file_size > self.MAX_IMAGE_FILE_SIZE:
                    raise ValueError(
                        f"Image {os.path.basename(file_path)} exceeds size limit of 5MB"
                    )
            else:
                if file_size > self.MAX_TEXT_FILE_SIZE:
                    raise ValueError(
                        f"File {os.path.basename(file_path)} exceeds size limit of 10MB"
                    )
                
                # Check word count for text files
                word_count = await self._count_words(file_path)
                if word_count > self.MAX_WORD_COUNT:
                    raise ValueError(
                        f"File {os.path.basename(file_path)} exceeds word limit of {self.MAX_WORD_COUNT}"
                    )
        
        # Check combined size limit
        if combined_size > self.MAX_COMBINED_SIZE:
            raise ValueError("Combined file size exceeds limit of 10MB")
    
    def validate_file_input(
        self,
        file_input: Optional[Union[str, List[str], List[Any]]]
    ) -> Optional[List[str]]:
        """
        Validate and process file input into a consistent format.
        Handles various input types including Gradio file objects.
        
        Args:
            file_input: Raw file input (string, list, or Gradio file objects)
            
        Returns:
            List of validated file paths or None if no valid input
        """
        if not file_input:
            return None
            
        # Convert to list if single input
        if isinstance(file_input, (str, Path)):
            file_input = [file_input]
            
        # Process list of inputs
        processed_files = []
        for file_obj in file_input:
            try:
                # Handle different input types
                if isinstance(file_obj, (str, Path)):
                    file_path = str(file_obj)
                elif hasattr(file_obj, 'name'):  # Gradio file object
                    file_path = file_obj.name
                else:
                    logger.warning(f"Unsupported file input type: {type(file_obj)}")
                    continue
                
                # Validate file exists and type is supported
                if not Path(file_path).exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                    
                if not self.is_valid_file_type(file_path):
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                    
                processed_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error processing file input {file_obj}: {str(e)}")
                continue
                
        return processed_files if processed_files else None
    
    def is_valid_file_type(self, file_path: str) -> bool:
        """
        Validate file type against all supported formats.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Boolean indicating if file type is supported
        """
        ext = Path(file_path).suffix.lower()
        return ext in (
            self.supported_audio_formats |
            self.supported_text_formats |
            self.supported_image_formats
        )

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of all supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        all_formats = (
            self.supported_audio_formats |
            self.supported_text_formats |
            self.supported_image_formats
        )
        return sorted(list(all_formats))
    
    async def _count_words(self, file_path: str) -> int:
        """Count words in a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split on whitespace and filter out empty strings
                words = [w for w in content.split() if w.strip()]
                return len(words)
        except Exception as e:
            logger.error(f"Error counting words in {file_path}: {str(e)}")
            raise
    
    async def load_documents_with_validation(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None
    ) -> List[Document]:
        """Load documents with validation before processing."""
        if file_paths:
            # Normalize to list
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            
            # Validate files before processing
            await self.validate_files(file_paths)
        
        # Use existing method for loading
        return self.load_documents(file_paths, urls)
    
    # ----------------------
    # Document Type-Specific Loading Functions
    # ----------------------
    
    def _load_text_document(self, file_path: str, file_extension: str) -> List[Document]:
        """
        Load text-based documents (txt, pdf, docx).
        
        Args:
            file_path: Path to the file
            file_extension: Extension of the file
            
        Returns:
            List of Document objects
        """
        try:
            if file_extension == '.txt':
                return TextLoader(file_path).load()
            elif file_extension == '.py':
                return TextLoader(file_path).load()
            elif file_extension == '.md':
                return UnstructuredMarkdownLoader(file_path).load()
            elif file_extension == '.pdf':
                return self._load_pdf(file_path)
            elif file_extension == '.docx':
                return Docx2txtLoader(file_path).load()
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error loading text document {file_path}: {str(e)}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF document with OCR fallback.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        docs = []
        pdf = fitz.open(file_path)
        
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            
            # Only use OCR if no text is extracted and OCR is enabled
            if not text.strip() and self.perform_ocr:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1
                    }
                ))
                
        return docs
    
    def _load_image_document(self, file_path: str) -> List[Document]:
        """
        Load image file and extract metadata and text via OCR if enabled.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of Document objects
        """
        try:
            mime_type = mimetypes.guess_type(file_path)[0]
            file_name = Path(file_path).name
            
            # Basic metadata for the image
            metadata = {
                "source": str(file_path),
                "type": "image",
                "mime_type": mime_type,
                "file_name": file_name,
                "has_ocr": False
            }
            
            documents = []
            
            # Create document for image metadata
            img_doc = Document(
                page_content=f"Image file: {file_name}",
                metadata=metadata.copy()
            )
            documents.append(img_doc)
            
            # Perform OCR if enabled
            if self.perform_ocr:
                try:
                    with Image.open(file_path) as img:
                        # Convert image to RGB if necessary
                        if img.mode not in ('L', 'RGB'):
                            img = img.convert('RGB')
                        
                        # Get image dimensions
                        width, height = img.size
                        metadata.update({
                            "width": width,
                            "height": height,
                            "mode": img.mode
                        })
                        
                        # Perform OCR
                        ocr_text = pytesseract.image_to_string(img)
                        
                        if ocr_text.strip():
                            # Create document for OCR text
                            ocr_metadata = metadata.copy()
                            ocr_metadata.update({
                                "content_type": "ocr_text",
                                "has_ocr": True
                            })
                            ocr_doc = Document(
                                page_content=ocr_text.strip(),
                                metadata=ocr_metadata
                            )
                            documents.append(ocr_doc)
                            
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for {file_path}: {str(ocr_error)}")
                    # Still return the image document even if OCR fails
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return []
    
    def _load_audio_document(self, file_path: str) -> List[Document]:
        """
        Process a single audio file for later transcription.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            List of Document objects with audio metadata
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_audio_formats:
                raise ValueError(f"Unsupported audio format: {file_extension}")

            processed_path = self._prepare_audio_for_processing(file_path)
            
            return [Document(
                page_content="Audio file prepared for transcription",
                metadata={
                    "source": file_path,
                    "processed_path": processed_path,
                    "file_type": file_extension[1:],
                    "original_path": file_path,
                    "sample_rate": self.audio_sample_rate,
                    "content_type": "audio"
                }
            )]

        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            return []
    
    def _prepare_audio_for_processing(self, file_path: str) -> str:
        """
        Prepare audio file for processing (standardize format).
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Path to the processed audio file
        """
        try:
            audio = AudioSegment.from_file(file_path)
            
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            if audio.frame_rate != self.audio_sample_rate:
                audio = audio.set_frame_rate(self.audio_sample_rate)
            
            output_path = self.temp_dir / f"{Path(file_path).stem}_processed.wav"
            
            audio.export(
                output_path,
                format="wav",
                parameters=["-ac", "1", "-ar", str(self.audio_sample_rate)]
            )
            
            return str(output_path)

        except Exception as e:
            logger.error(f"Error preparing audio file {file_path}: {str(e)}")
            raise
    
    # ----------------------
    # URL Loading Functions
    # ----------------------
    
    def _load_from_urls(self, urls: str) -> List[Document]:
        """
        Load documents from URLs.
        
        Args:
            urls: String containing URLs (one per line)
            
        Returns:
            List of Document objects
        """
        docs = []
        for url in urls.split('\n'):
            url = url.strip()
            if url:
                try:
                    # Check if it's a video platform URL
                    if self._is_video_platform_url(url):
                        audio_path = self._download_platform_audio(url)
                        if audio_path:
                            docs.extend(self._load_audio_document(audio_path))
                    else:
                        # Regular web page
                        loaded_docs = WebBaseLoader(url).load()
                        docs.extend(loaded_docs)
                except Exception as e:
                    logger.error(f"Error loading URL {url}: {str(e)}")
        return docs

    def _load_from_files(self, file_paths: Union[str, List[str], List[Any]]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: Single file path or list of file paths
            
        Returns:
            List of Document objects
        """
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        
        docs = []
        for file_obj in file_paths:
            try:
                file_path = file_obj if isinstance(file_obj, str) else file_obj.name
                docs.extend(self.load_document(file_path))
                    
            except Exception as e:
                logger.error(f"Error processing file {file_obj}: {str(e)}")
                
        if not docs:
            raise ValueError("No documents were successfully loaded.")
            
        return docs
    
    def _is_video_platform_url(self, url: str) -> bool:
        """
        Check if URL is from supported video platforms.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if URL is from a supported video platform
        """
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in [
            'youtube.com', 'youtu.be', 'www.youtube.com',
            'vimeo.com', 'www.vimeo.com',
            'tiktok.com', 'www.tiktok.com', 'novulo.com' 
        ])

    def _download_platform_audio(self, url: str) -> Optional[str]:
        """
        Download audio from video platforms.
        
        Args:
            url: URL to the video
            
        Returns:
            Path to the downloaded audio file
        """
        try:
            output_path = self.temp_dir / f"video_{hash(url)}.wav"
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': str(output_path.with_suffix('')),
                'verbose': True,
                'no_warnings': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Attempting to download: {url}")
                ydl.download([url])
                
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading platform audio: {str(e)}")
            raise
            
    async def download_url_content(self, url: str) -> Optional[str]:
        """
        Download content from a URL asynchronously.
        
        Args:
            url: URL to download
            
        Returns:
            Path to the downloaded file
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url.lstrip('/')
                
            if self._is_video_platform_url(url):
                return await asyncio.to_thread(self._download_platform_audio, url)
            
            # Regular file download
            response = await asyncio.to_thread(requests.get, url, stream=True)
            response.raise_for_status()
            
            filename = Path(urlparse(url).path).name or "downloaded_file"
            temp_path = self.temp_dir / filename
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error downloading from URL {url}: {str(e)}")
            return None
    
    # ----------------------
    # Cleanup Functions
    # ----------------------
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")