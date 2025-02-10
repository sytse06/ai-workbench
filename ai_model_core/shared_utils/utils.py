# ai_model_core/shared_utils/utils.py
# Standard library imports
from pathlib import Path
from typing import (
    List,
    Generator,
    Literal,
    Any,
    Optional,
    Union,
    Tuple,
    Dict,
    Generator, 
    AsyncGenerator
)
import gradio as gr
import os
import logging
import mimetypes
import tempfile
from urllib.parse import urlparse, parse_qs

# Third-party imports
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    Docx2txtLoader, 
    UnstructuredMarkdownLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydub import AudioSegment
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from urllib.parse import urlparse
import requests
import yt_dlp

# Local imports
from ..config.settings import load_config

logger = logging.getLogger(__name__)

class EnhancedContentLoader:
    """
    A versatile content loader that handles multiple file types including text, PDFs, 
    audio, images and URLs with appropriate preprocessing and document splitting capabilities.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "input/tmp",
        audio_sample_rate: int = 16000,
        perform_ocr: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir = Path(temp_dir)
        self.audio_sample_rate = audio_sample_rate
        self.perform_ocr = perform_ocr
        
        #Supported formats
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

    def load_documents(
        self,
        file_paths: Optional[Union[str, List[str], List[Any]]] = None,
        urls: Optional[str] = None
    ) -> List[Document]:
        """
        Main method to load documents from various sources.
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

    def _load_from_urls(self, urls: str) -> List[Document]:
        """Load documents from URLs."""
        docs = []
        for url in urls.split('\n'):
            url = url.strip()
            if url:
                try:
                    loaded_docs = WebBaseLoader(url).load()
                    docs.extend(loaded_docs)
                except Exception as e:
                    logger.error(f"Error loading URL {url}: {str(e)}")
        return docs

    def _load_from_files(self, file_paths: Union[str, List[str], List[Any]]) -> List[Document]:
        """Load documents from file paths."""
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        
        docs = []
        for file_obj in file_paths:
            try:
                file_path = file_obj if isinstance(file_obj, str) else file_obj.name
                file_extension = Path(file_path).suffix.lower()
                
                # Handle different file types
                if file_extension in self.supported_text_formats:
                    loaded_docs = self._load_text_document(file_path, file_extension)
                elif file_extension in self.supported_audio_formats:
                    loaded_docs = self._process_audio_file(file_path)
                elif file_extension in self.supported_image_formats:
                    loaded_docs = self._process_image_file(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    continue
                    
                # Only extend docs if documents were actually loaded
                if loaded_docs:
                    docs.extend(loaded_docs)
                    
            except Exception as e:
                logger.error(f"Error processing file {file_obj}: {str(e)}")
                
        if not docs:
            raise ValueError("No documents were successfully loaded.")
            
        return docs
    
    def _process_image_file(self, file_path: str) -> List[Document]:
        """
        Process image files for both visual and textual content.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of Document objects containing image information and OCR text
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

    def is_valid_file_type(self, file_path: str) -> bool:
        """Validate file type against all supported formats."""
        ext = Path(file_path).suffix.lower()
        return ext in (
            self.supported_audio_formats |
            self.supported_text_formats |
            self.supported_image_formats
        )

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        all_formats = (
            self.supported_audio_formats |
            self.supported_text_formats |
            self.supported_image_formats
        )
        return sorted(list(all_formats))

    def _load_text_document(self, file_path: str, file_extension: str) -> List[Document]:
        """Load text-based documents (txt, pdf, docx)."""
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
        """Load PDF document with OCR fallback."""
        docs = []
        pdf = fitz.open(file_path)
        
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            
            # Only use OCR if no text is extracted
            if not text.strip():
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

    def split_documents(self, docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Split documents into smaller chunks."""
        try:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            return text_splitter.split_documents(docs)
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise

    def load_and_split_documents(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """Load and split documents from various sources."""
        try:
            docs = self.load_documents(file_paths, urls)
            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap
            return self.split_documents(docs, chunk_size, chunk_overlap)
        except Exception as e:
            logger.error(f"Error in load_and_split_documents: {str(e)}")
            raise

    def _process_audio_file(self, file_path: str) -> List[Document]:
        """Process a single audio file for transcription."""
        try:
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_audio_formats:
                raise ValueError(f"Unsupported audio format: {file_extension}")

            processed_path = self._prepare_audio_for_whisper(file_path)
            
            return [Document(
                page_content="Audio file prepared for transcription",
                metadata={
                    "source": file_path,
                    "processed_path": processed_path,
                    "file_type": file_extension[1:],
                    "original_path": file_path,
                    "sample_rate": self.audio_sample_rate
                }
            )]

        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            return []

    def _prepare_audio_for_whisper(self, file_path: str) -> str:
        """Prepare audio file for Whisper transcription."""
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

    def preprocess_audio(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None
    ) -> List[Document]:
        """Preprocess audio files from various sources."""
        docs = []
        
        try:
            if urls:
                for url in urls.split('\n'):
                    url = url.strip()
                    if url:
                        temp_path = self._download_audio_file(url)
                        if temp_path:
                            docs.extend(self._process_audio_file(temp_path))

            if file_paths:
                if isinstance(file_paths, str):
                    file_paths = [file_paths]
                
                for file_path in file_paths:
                    docs.extend(self._process_audio_file(file_path))

            return docs

        except Exception as e:
            logger.error(f"Error in preprocess_audio: {str(e)}")
            raise

    def _download_audio_file(self, url: str) -> Optional[str]:
        """Download audio file from URL with platform support."""
        try:
            if self._is_video_platform_url(url):
                return self._download_platform_audio(url)
                
            # Regular audio file download logic
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filename = Path(urlparse(url).path).name or "audio_file"
            temp_path = self.temp_dir / filename
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return str(temp_path)

        except Exception as e:
            logger.error(f"Error downloading audio file from {url}: {str(e)}")
            return None
    
    def _is_video_platform_url(self, url: str) -> bool:
        """Check if URL is from supported video platforms."""
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in [
            'youtube.com', 'youtu.be', 'www.youtube.com',
            'vimeo.com', 'www.vimeo.com',
            'tiktok.com', 'www.tiktok.com', 'novulo.com' 
        ])

    def _download_platform_audio(self, url: str) -> Optional[str]:
        """Download audio from video platforms."""
        try:
            output_path = self.temp_dir / f"video_{hash(url)}.wav"
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': str(output_path.with_suffix('')),
                'verbose': True,  # Add verbose output for debugging
                'no_warnings': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Attempting to download: {url}")
                ydl.download([url])
                
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading platform audio: {str(e)}")
            raise
    
    def process_files_for_context(
        self,
        documents: List[Document]
    ) -> Dict[str, List[str]]:
        """
        Organize processed documents by content type and prepare them for context usage.
        
        Args:
            documents: List of processed Document objects
            
        Returns:
            Dict containing organized content by type:
            {
                "text_content": List of text content from documents
                "image_descriptions": List of image descriptions and OCR text
                "audio_content": List of audio processing results
                "other_content": List of other content types
            }
        """
        context = {
            "text_content": [],
            "image_descriptions": [],
            "audio_content": [],
            "other_content": []
        }
        
        for doc in documents:
            try:
                # Get file extension from source if available
                source = doc.metadata.get('source', '')
                file_extension = Path(source).suffix.lower() if source else ''
                
                # Process based on content type
                if doc.metadata.get('type') == 'image':
                    if doc.metadata.get('has_ocr', False):
                        context['image_descriptions'].append(
                            f"OCR text from {doc.metadata['file_name']}: {doc.page_content}"
                        )
                    else:
                        context['image_descriptions'].append(
                            f"Image file {doc.metadata['file_name']}: {doc.page_content}"
                        )
                elif file_extension in self.supported_text_formats:
                    context['text_content'].append(
                        f"Content from {Path(source).name}:\n{doc.page_content}"
                    )
                elif file_extension in self.supported_audio_formats:
                    context['audio_content'].append(
                        f"Audio file {Path(source).name}: {doc.page_content}"
                    )
                else:
                    context['other_content'].append(doc.page_content)
                    
            except Exception as e:
                logger.error(f"Error processing document for context: {str(e)}")
                continue
                
        return context

    def load_and_process_files(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        return_raw_documents: bool = False
    ) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], List[Document]]]:
        """
        Load, split, and organize documents from various sources.
        
        Args:
            file_paths: Single file path or list of file paths
            urls: String containing URLs (one per line)
            chunk_size: Optional custom chunk size for splitting
            chunk_overlap: Optional custom chunk overlap for splitting
            return_raw_documents: If True, also returns the raw document list
            
        Returns:
            If return_raw_documents is False:
                Dict containing organized content by type
            If return_raw_documents is True:
                Tuple of (organized content dict, raw document list)
        """
        try:
            # Load and split documents
            documents = self.load_and_split_documents(
                file_paths=file_paths,
                urls=urls,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Process documents into organized context
            context = self.process_files_for_context(documents)
            
            if return_raw_documents:
                return context, documents
            return context
            
        except Exception as e:
            logger.error(f"Error in load_and_process_files: {str(e)}")
            empty_context = {
                "text_content": [],
                "image_descriptions": [],
                "audio_content": [],
                "other_content": []
            }
            if return_raw_documents:
                return empty_context, []
            return empty_context

    def get_formatted_context(
        self,
        context: Dict[str, List[str]],
        include_types: Optional[List[str]] = None,
        format_type: Literal["string", "list", "dict"] = "string"
    ) -> Union[str, List[str], Dict[str, List[str]]]:
        """
        Format the processed context in the desired output format.
        
        Args:
            context: Dict containing organized content by type
            include_types: Optional list of content types to include
            format_type: Desired output format ("string", "list", or "dict")
            
        Returns:
            Formatted context in the specified format
        """
        if include_types is None:
            include_types = ["text_content", "image_descriptions", "audio_content", "other_content"]
            
        # Filter context based on included types
        filtered_context = {
            k: v for k, v in context.items() 
            if k in include_types and v
        }
        
        if format_type == "dict":
            return filtered_context
            
        # Flatten content into list
        flattened_content = []
        for content_type, content_list in filtered_context.items():
            if content_list:
                flattened_content.extend(content_list)
                
        if format_type == "list":
            return flattened_content
            
        # Return as formatted string
        return "\n\n".join(flattened_content) if flattened_content else ""
          
    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file_path in self.temp_dir.glob('*'):
                file_path.unlink()
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
