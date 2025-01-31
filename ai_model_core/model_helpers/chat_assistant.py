# model_helpers/chat_assistant.py
# Standard library imports
import logging
from pathlib import Path
import asyncio
import os
import base64
from io import BytesIO
from PIL import Image
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

# Third-party imports
from langchain.schema import (
    HumanMessage,
    AIMessage,
    Document,
    BaseMessage,
    SystemMessage
)
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredMarkdownLoader, 
    Docx2txtLoader
)
import gradio as gr

# Local imports
from ai_model_core.shared_utils.factory import (
    get_model,
    update_model
)
from ai_model_core.config.settings import (
    load_config
)
from ai_model_core.shared_utils.utils import (
    EnhancedContentLoader,
    get_prompt_template,
    get_system_prompt,
    _format_history,
    format_assistant_message,
    format_user_message,
    format_file_content,
    convert_history_to_messages
)

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "AI-Workbench/1.0"

logger = logging.getLogger(__name__)

class ChatAssistant:
    #File upload constraints
    MAX_TEXT_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    MAX_IMAGE_FILE_SIZE = 5 * 1024 * 1024   # 5MB in bytes
    MAX_WORD_COUNT = 4000
    MAX_COMBINED_SIZE = 10 * 1024 * 1024    # 10MB total
    
    def __init__(
        self, 
        model_choice: str, 
        temperature: float = 0.7, 
        max_tokens: int = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temp_dir: str = "input/tmp"
    ):
        """
        Initialize ChatAssistant with enhanced capabilities for chat, prompt, and file handling.
        
        Args:
            model_choice: Name of the model to use
            temperature: Temperature for text generation
            max_tokens: Maximum tokens in response
            chunk_size: Size of text chunks for document processing
            chunk_overlap: Overlap between consecutive chunks
            temp_dir: Directory for temporary files
        """
        self.model = get_model(model_choice)
        self.model_choice = model_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.documents: List[Document] = []
        self.config = load_config()
        
        # Initialize the content loader
        self.content_loader = EnhancedContentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temp_dir=temp_dir
        )
        
    async def update_model(self, model_choice: str):
        if self.model_choice != model_choice:  # Add 4 spaces indentation
            new_model = await update_model(model_choice, self.model_choice)
            if new_model:
                self.model = new_model 
                self.model_choice = model_choice
        
    async def process_chat_context_files(
        self, 
        files: List[gr.File]
    ) -> List[Document]:
        """Process and validate uploaded files for chat context."""
        if not files:
            return []

        try:
            await self._validate_files(files)
            
            # Process files concurrently
            tasks = []
            for file in files:
                file_path = file.name if hasattr(file, "name") else str(file)
                tasks.append(self._process_single_file(file_path))
            
            # Gather results
            results = await asyncio.gather(*tasks)
            return [doc for docs in results for doc in docs]  # Flatten list

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise
        
    async def _validate_files(self, files: List[gr.File]) -> None:
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
        
    async def detect_file_type(self, file: gr.File) -> str:
        """Detect if a file is an image or text based on its extension."""
        file_path = file.name if hasattr(file, "name") else str(file)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        if Path(file_path).suffix.lower() in image_extensions:
            return "image"
        return "text"

    async def _process_single_file(self, file_path: str) -> List[Document]:
        """Process a single file and return documents."""
        try:
            if self._is_image_file(file_path):
                # Handle image files
                content = await self._process_image(file_path)
                return [Document(
                    page_content=content,
                    metadata={"source": file_path, "type": "image"}
                )]
            else:
                # Handle text files
                return self._load_text_document(file_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def _is_image_file(self, file_path: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        return Path(file_path).suffix.lower() in image_extensions

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

    async def _process_image(self, file_path: str) -> str:
        """Process image file and return content description."""
        try:
            # If you have image processing capabilities, implement them here
            # For now, return basic image metadata
            image = Image.open(file_path)
            return f"Image file: {os.path.basename(file_path)}, " \
                   f"Size: {image.size}, Mode: {image.mode}"
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            raise

    def _load_text_document(self, file_path: str) -> List[Document]:
        """Load a text document with appropriate loader based on file type."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt' or file_extension == '.py':
                return TextLoader(file_path).load()
            elif file_extension == '.md':
                return UnstructuredMarkdownLoader(file_path).load()
            elif file_extension == '.pdf':
                return self._load_pdf(file_path)
            elif file_extension == '.docx':
                return Docx2txtLoader(file_path).load()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error loading text document {file_path}: {str(e)}")
            raise

    async def load_documents(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        urls: Optional[str] = None
    ) -> bool:
        """Load documents from files or URLs."""
        try:
            new_docs = self.content_loader.load_documents(file_paths=file_paths, urls=urls)
            if new_docs:
                self.documents.extend(new_docs)
                logger.info(f"Successfully loaded {len(new_docs)} new documents")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False

    def get_context_from_docs(
        self,
        message: str,
        use_context: bool = True,
        max_documents: int = 3
    ) -> str:
        """Get relevant context from loaded documents."""
        if not use_context or not self.documents:
            return ""
            
        return self.get_relevant_context(message, max_documents)
    
    def _convert_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _process_message_content(self, content: List[Union[str, Dict]]) -> Tuple[str, Optional[Image.Image]]:
        """Process Gradio v5 message content to extract text and images."""
        message_text = ""
        image = None

        for item in content:
            if isinstance(item, str):
                message_text += item + " "
            elif isinstance(item, dict):
                if "path" in item:  # Image file
                    try:
                        image = Image.open(item["path"])
                    except Exception as e:
                        logger.error(f"Error loading image: {e}")
                        
        return message_text.strip(), image

    def get_relevant_context(self, message: str, max_documents: int = 3) -> str:
        """Retrieve relevant context based on the message."""
        if not self.documents:
            return ""
            
        relevant_docs = []
        message_words = set(message.lower().split())
        
        for doc in self.documents:
            content = doc.page_content.lower()
            relevance_score = sum(1 for word in message_words if word in content)
            if relevance_score > 0:
                relevant_docs.append((relevance_score, doc))
        
        relevant_docs.sort(reverse=True, key=lambda x: x[0])
        selected_docs = relevant_docs[:max_documents]
        
        if not selected_docs:
            return ""
            
        context_parts = []
        for _, doc in selected_docs:
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ''
            context_parts.append(f"From {source}{page_info}:\n{doc.page_content}\n")
            
        return "\n".join(context_parts)

    async def chat(
        self,
        message: Dict,  # Gradio v5 message format
        history: List[Dict],
        history_flag: bool = True,
        stream: bool = False,
        use_context: bool = True,
        prompt_info: Optional[str] = None,
        language_choice: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Unified chat function handling both text and vision inputs."""
        try:
            # Process message content
            message_text, image = self._process_message_content(message["content"])
            
            # Prepare base messages
            messages = []
            
            # Add system message if using prompts
            if prompt_info and language_choice:
                system_prompt = get_system_prompt(language_choice, self.config)
                messages.append(SystemMessage(content=system_prompt))
                
                # Format message with prompt template
                prompt_template = get_prompt_template(prompt_info, self.config)
                message_text = prompt_template.format(
                    prompt_info=prompt_info,
                    user_message=message_text
                )

            # Add history if enabled
            if history_flag and history:
                messages.extend(_format_history(history))

            # Handle image content
            if image:
                image_b64 = self._convert_to_base64(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        }
                    ]
                })
            else:
                messages.append(HumanMessage(content=message_text))

            # Add relevant context if enabled
            if use_context:
                context = self.get_context_from_docs(message_text)
                if context:
                    # Insert context before the user's message
                    messages.insert(-1, SystemMessage(content=f"Context:\n{context}"))

            # Configure model parameters
            self._configure_model(stream)

            # Generate response
            if stream:
                async for chunk in self.model.astream(messages):
                    yield chunk.content
            else:
                result = await self.model.agenerate([messages])
                yield result.generations[0][0].text

        except Exception as e:
            logger.error(f"Error in chat generation: {str(e)}")
            yield f"An error occurred: {str(e)}"

    def _configure_model(self, stream: bool = False):
        """Configure model parameters based on model type."""
        if "ollama" in self.model_choice.lower():
            self.model = self.model.bind(
                stop=None,
                stream=stream
            )
        elif "gemini" in self.model_choice.lower():
            self.model = self.model.bind(
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
        else:
            self.model = self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
    def clear_documents(self):
        """Clear all loaded documents."""
        self.documents = []
        logger.info("Cleared all documents from memory")

    def get_document_summary(self) -> str:
        """Get a summary of loaded documents."""
        if not self.documents:
            return "No documents currently loaded."
            
        summary_parts = []
        sources = {}
        
        for doc in self.documents:
            source = doc.metadata.get('source', 'Unknown source')
            sources[source] = sources.get(source, 0) + 1
                
        summary_parts.append(f"Total documents loaded: {len(self.documents)}")
        summary_parts.append("\nDocuments by source:")
        for source, count in sources.items():
            summary_parts.append(f"- {source}: {count} document(s)")
            
        return "\n".join(summary_parts)

    def set_temperature(self, temperature: float):
        """Set the temperature parameter."""
        self.temperature = temperature

    def set_max_tokens(self, max_tokens: int):
        """Set the max_tokens parameter."""
        self.max_tokens = max_tokens