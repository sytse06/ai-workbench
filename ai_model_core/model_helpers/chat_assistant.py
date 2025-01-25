# model_helpers/chat_assistant.py
# Standard library imports
import logging
from pathlib import Path
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
        
    async def process_chat_context_files(self, files: List[gr.File]) -> List[Document]:
        if not files:
            return []
        try:
            file_paths = [f.name for f in files]
            await self.load_documents(file_paths=file_paths)
            return self.documents
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return []

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