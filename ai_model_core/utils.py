# ai_model_interface/utils.py
from pathlib import Path
from typing import List, Union, Any, Optional
from langchain.schema import Document
from langchain.document_loaders import TextLoader, WebBaseLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydub import AudioSegment
import fitz  # PyMuPDF
import logging
import tempfile
from PIL import Image
import pytesseract
from urllib.parse import urlparse
import requests
import os

logger = logging.getLogger(__name__)

class EnhancedContentLoader:
    """
    A versatile content loader that handles multiple file types including text, PDFs, 
    audio, and URLs with appropriate preprocessing and document splitting capabilities.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "input/tmp",
        audio_sample_rate: int = 16000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir = Path(temp_dir)
        self.audio_sample_rate = audio_sample_rate
        self.supported_audio_formats = {'.mp4', '.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        self.supported_text_formats = {'.txt', '.pdf', '.docx'}
        
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
                docs.extend(url_docs)

            # Process files if provided
            if file_paths:
                file_docs = self._load_from_files(file_paths)
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
                
                # Handle text documents first
                if file_extension in self.supported_text_formats:
                    docs.extend(self._load_text_document(file_path, file_extension))
                # Then handle audio files
                elif file_extension in self.supported_audio_formats:
                    docs.extend(self._process_audio_file(file_path))
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_obj}: {str(e)}")
                
        return docs

    def _load_text_document(self, file_path: str, file_extension: str) -> List[Document]:
        """Load text-based documents (txt, pdf, docx)."""
        try:
            if file_extension == '.txt':
                return TextLoader(file_path).load()
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
        """Download audio file from URL."""
        try:
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

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file_path in self.temp_dir.glob('*'):
                file_path.unlink()
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
                
def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config["system_prompt_settings"][language_choice]["system_prompt"]
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt_template(prompt_info: str, config: dict, language_choice: str = "english") -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate using the prompt_info and config.

    :param prompt_info: The prompt info selected by the user
    :param config: The loaded configuration
    :return: ChatPromptTemplate
    """
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    prompt_text = config['prompts'].get(prompt_info, "")
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{prompt_info}\n\n{user_message}")
    ])

def _format_history(history: List[tuple[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history

# Function to load config from a YAML file
def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config