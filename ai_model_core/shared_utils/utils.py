# ai_model_core/shared_utils/utils.py
# Standard library imports
from pathlib import Path
from typing import (
    List,
    Generator,
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
from langchain.schema import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage
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
from ai_model_core.config.settings import load_config

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
        self.supported_text_formats = {'.txt', '.pdf', '.docx', '.md', '.py'}
        
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
def get_prompt(prompt_name: str) -> str:
    config = load_config()
    return config['prompts'].get(prompt_name)

def get_prompt_list(language: str) -> List[str]:
    config = load_config()
    prompts = config.get("prompts", {})
    return prompts.get(language, [])

def update_prompt_list(language: str):
    new_prompts = get_prompt_list(language)
    return gr.Dropdown(choices=new_prompts)

# Functions to support new messaging format Gradiov5
async def format_user_message(
    message: str, 
    history: List[Dict] = None, 
    files: List = None
) -> Tuple[str, List[Dict]]:
    """
    Format a user message, optionally including file attachments.
    
    Args:
        message: User's text message
        history: Optional chat history
        files: Optional list of file objects
        
    Returns:
        Tuple of (empty string, new history list)
    """
    if history is None:
        history = []
        
    new_history = history.copy()

    if files:
        # If we have files, create a list of content items
        content = []
        if message:
            content.append(message.strip())
            
        for file in files:
            file_path = file.name if hasattr(file, 'name') else str(file)
            file_ext = Path(file_path).suffix.lower()
            
            # Handle different file types
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                content.append({
                    "type": "image",
                    "path": file_path,
                    "alt_text": f"Image: {os.path.basename(file_path)}"
                })
            else:
                content.append({
                    "type": "file",
                    "path": file_path,
                    "alt_text": f"File: {os.path.basename(file_path)}"
                })
                
        new_history.append({"role": "user", "content": content})
    elif message:
        # Simple text message
        new_history.append({"role": "user", "content": message.strip()})
    
    return "", new_history

def format_assistant_message(content: str, metadata: Dict = None) -> Dict:
    """
    Format a message from the assistant.
    
    Args:
        content: Message content
        metadata: Optional metadata dictionary
        
    Returns:
        Formatted message dictionary
    """
    message = {
        "role": "assistant",
        "content": content
    }
    if metadata:
        message["metadata"] = metadata
    return message

def format_file_content(file_path: str, alt_text: str = None, file_type: str = None) -> dict:
    """Format file content as a properly structured message."""
    if not alt_text:
        alt_text = f"File: {Path(file_path).name}"
        
    file_content = {
        "path": file_path,
        "alt_text": alt_text,
        "type": file_type or Path(file_path).suffix[1:]
    }
    
    return {
        "role": "user",
        "content": [file_content]
    }

def convert_history_to_messages(history: List[Union[BaseMessage, Dict, Tuple[str, str]]]) -> List[Dict]:
    """
    Convert different history formats to Gradio v5 messages format.
    
    Args:
        history: Chat history in various formats (LangChain messages, dicts, or tuples)
        
    Returns:
        List of messages in Gradio v5 format
    """
    messages = []
    
    if not history:
        return messages
        
    for entry in history:
        if isinstance(entry, (HumanMessage, AIMessage, SystemMessage)):
            # Handle LangChain message types
            role = {
                HumanMessage: "user",
                AIMessage: "assistant",
                SystemMessage: "system"
            }.get(type(entry))
            messages.append({
                "role": role,
                "content": entry.content
            })
        elif isinstance(entry, tuple):
            # Handle tuple format (user_msg, assistant_msg)
            user_msg, assistant_msg = entry
            if user_msg:  # Only add non-empty messages
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:  # Only add non-empty messages
                messages.append({"role": "assistant", "content": assistant_msg})
        elif isinstance(entry, dict):
            # Handle dictionary format
            if "role" in entry and "content" in entry:
                messages.append(entry.copy())  # Use copy to avoid modifying original
            elif "speaker" in entry:  # Alternative format
                messages.append({
                    "role": entry["speaker"],
                    "content": entry.get("message", "")
                })
            elif "type" in entry:  # Another alternative format
                messages.append({
                    "role": entry["type"],
                    "content": entry.get("text", "")
                })
    
    return messages

def _format_history(history: List[Union[Dict, Tuple[str, str]]]) -> List[BaseMessage]:
    """
    Convert chat history to LangChain message format.
    
    Args:
        history: List of messages in dictionary or tuple format
        
    Returns:
        List of LangChain message objects
    """
    messages = []
    
    if not history:
        return messages
        
    for entry in history:
        if isinstance(entry, tuple):
            # Handle tuple format (user_msg, assistant_msg)
            user_msg, assistant_msg = entry
            if user_msg:
                messages.append(HumanMessage(content=user_msg))
            if assistant_msg:
                messages.append(AIMessage(content=assistant_msg))
        elif isinstance(entry, dict):
            # Handle dictionary format
            role = entry.get("role", "").lower()
            content = entry.get("content", "")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
                
    return messages

async def process_message(
    message: Dict,
    history: List[Dict],
    model_choice: str,
    prompt_info: Optional[str] = None,
    language_choice: Optional[str] = None,
    history_flag: bool = True,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    files: Optional[List[gr.File]] = None,
    use_context: bool = True
) -> AsyncGenerator[Dict, None]:
    try:
        new_model = await update_model(model_choice, chat_assistant.model_choice)
        if new_model:
            chat_assistant.model = new_model
            chat_assistant.model_choice = model_choice

        result = []
        async for chunk in chat_assistant.chat(
            message=message,
            history=history,
            prompt_info=prompt_info,
            language_choice=language_choice,
            history_flag=history_flag,
            stream=True,
            use_context=use_context
        ):
            result.append(chunk)
            # Format each chunk as a proper message
            yield format_assistant_message(''.join(result))
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        yield format_assistant_message(f"An error occurred: {str(e)}")