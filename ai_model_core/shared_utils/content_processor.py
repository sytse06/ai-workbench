# ai_model_core/shared_utils/content_processor.py
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple, Callable, AsyncGenerator
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import os

from langchain.schema import Document

# Local imports
from .content_loader import EnhancedContentLoader

logger = logging.getLogger(__name__)

class AssistantType(Enum):
    """
    Enum defining different types of assistants that can process content.
    
    Types:
    CHAT: Basic chat assistant using LLMs
    RAG: Retrieval-augmented generation assistant
    SUMMARIZATION: Document summarization assistant
    TRANSCRIPTION: Audio/video transcription assistant
    AGENTS: Task-specific agent-based assistants
    """
    CHAT = "chat"
    RAG = "rag"
    SUMMARIZATION = "summarization"
    TRANSCRIPTION = "transcription"
    AGENTS = "agents"

@dataclass
class LoaderConfig:
    """Configuration for document loading and processing."""
    chunk_size: int
    chunk_overlap: int
    process_callback: Optional[Callable] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

class BaseContentProcessor:
    """
    Base class for assistant-specific content processing.
    
    This class defines the interface for processing content after it has been loaded.
    Each assistant type should have its own processor implementation.
    """
    
    def __init__(self, loader: Optional[EnhancedContentLoader] = None):
        """
        Initialize the content processor.
        
        Args:
            loader: Optional EnhancedContentLoader instance
        """
        self.loader = loader or EnhancedContentLoader()
    
    async def process_documents(self, 
                               documents: List[Document], 
                               **kwargs) -> Tuple[str, Any]:
        """
        Process documents for a specific assistant type.
        
        Args:
            documents: List of Document objects
            kwargs: Additional processing parameters
            
        Returns:
            Tuple of (status message, processed result)
        """
        raise NotImplementedError("Subclasses must implement process_documents")
    
    def process_image_document(self, document: Document, for_vlm: bool = False) -> Dict[str, Any]:
        """
        Process an image document for different use cases.
        
        Args:
            document: Image document to process
            for_vlm: Whether this is for a vision language model
            
        Returns:
            Dictionary with processed image information
        """
        metadata = document.metadata
        file_path = metadata.get("source", "")
        filename = metadata.get("file_name", Path(file_path).name if file_path else "Unknown image")
        
        result = {
            "filename": filename,
            "path": file_path,
            "type": "image",
            "has_ocr": metadata.get("has_ocr", False),
            "ocr_text": document.page_content if metadata.get("has_ocr", False) else "",
            "width": metadata.get("width", "unknown"),
            "height": metadata.get("height", "unknown"),
            "for_vlm": for_vlm
        }
        
        return result
        
    def get_image_context(self, documents: List[Document], max_images: int = 3, for_vlm: bool = False) -> Tuple[str, List[str]]:
        """
        Create formatted context from image documents.
        
        Args:
            documents: List of image documents
            max_images: Maximum number of images to include
            for_vlm: Whether this is for a vision language model
            
        Returns:
            Tuple of (formatted context string, list of image paths)
        """
        if not documents:
            return "", []
        
        # Filter to just image documents
        image_docs = [doc for doc in documents if doc.metadata.get("type") == "image"]
        if not image_docs:
            return "", []
        
        # Take most recent images first, up to max_images
        selected_docs = image_docs[-max_images:]
        
        # Process images
        context_parts = []
        image_paths = []
        
        for doc in selected_docs:
            img_info = self.process_image_document(doc, for_vlm)
            
            if img_info["path"] and os.path.exists(img_info["path"]):
                image_paths.append(img_info["path"])
                
                # Format context differently based on OCR availability
                if img_info["has_ocr"] and img_info["ocr_text"]:
                    context_parts.append(
                        f"Image: {img_info['filename']} (Dimensions: {img_info['width']}x{img_info['height']})\n"
                        f"Text content: {img_info['ocr_text']}"
                    )
                else:
                    context_parts.append(
                        f"Image: {img_info['filename']} (Dimensions: {img_info['width']}x{img_info['height']})"
                    )
        
        return "\n\n".join(context_parts), image_paths
    
    def get_metadata_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate a summary of document metadata.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with metadata summary
        """
        if not documents:
            return {"count": 0, "types": {}}
        
        types = {}
        sources = set()
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type in types:
                types[doc_type] += 1
            else:
                types[doc_type] = 1
                
            source = doc.metadata.get("source", "")
            if source:
                sources.add(source)
        
        return {
            "count": len(documents),
            "types": types,
            "unique_sources": len(sources),
            "sources": list(sources)
        }
    
    def format_files_message(self, documents: List[Document]) -> str:
        """
        Generate a summary message for files loaded to display in chat UI.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted message about loaded files
        """
        if not documents:
            return "No files processed."
            
        summary = self.get_metadata_summary(documents)
        
        # Get file counts by type
        file_counts = []
        for doc_type, count in summary["types"].items():
            file_counts.append(f"{count} {doc_type} file(s)")
            
        # Create message
        if len(summary["sources"]) > 3:
            source_text = f"{len(summary['sources'])} files"
        else:
            source_text = ", ".join([Path(s).name for s in summary["sources"]])
            
        return f"📁 Processed {source_text}: {', '.join(file_counts)}"

class ChatContentProcessor(BaseContentProcessor):
    """
    Process content specifically for chat context.
    """
    
    async def process_documents(self, 
                               documents: List[Document], 
                               **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process documents for chat context.
        
        Args:
            documents: List of Document objects
            kwargs: Additional processing parameters
                - query: Optional user query for relevance-based formatting
                - max_documents: Maximum number of documents to include
                
        Returns:
            Tuple of (status message, processed context)
        """
        if not documents:
            return "No documents to process", {"context": "", "metadata": {}}
        
        query = kwargs.get("query", "")
        max_documents = kwargs.get("max_documents", 5)
        
        # Format context based on document types
        context = self.format_context_from_docs(documents, query, max_documents)
        metadata = self.get_metadata_summary(documents)
        files_message = self.format_files_message(documents)
        
        return (
            f"Processed {len(documents)} documents for chat context",
            {
                "context": context, 
                "metadata": metadata,
                "files_message": files_message
            }
        )
    
    def format_context_from_docs(self, 
                                documents: List[Document],
                                query: str = "", 
                                max_documents: int = 5) -> str:
        """
        Format documents into context string for chat with improved image handling.
        
        Args:
            documents: List of Document objects
            query: Optional user query for relevance-based formatting
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        # Group documents by type
        docs_by_type = {}
        for doc in documents:
            doc_type = doc.metadata.get("type", "text")
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append(doc)
        
        # Process each type with appropriate formatter
        context_parts = []
        
        # Process image documents first for better VLM handling
        if "image" in docs_by_type:
            image_context = self._format_image_documents(
                docs_by_type["image"], query, max_documents)
            if image_context:
                context_parts.append(image_context)
        
        # Process text documents
        if "text" in docs_by_type:
            text_context = self._format_text_documents(
                docs_by_type["text"], query, max_documents)
            if text_context:
                context_parts.append(text_context)
        
        # Process audio documents
        if "audio" in docs_by_type:
            audio_context = self._format_audio_documents(
                docs_by_type["audio"], max_documents)
            if audio_context:
                context_parts.append(audio_context)
        
        # Combine all formatted contexts
        return "\n\n".join(context_parts)
    
    def _format_text_documents(self, 
                              documents: List[Document],
                              query: str = "", 
                              max_documents: int = 5) -> str:
        """
        Format text documents, optionally prioritizing by relevance to query.
        
        Args:
            documents: List of Document objects
            query: User query for relevance-based formatting
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted text context
        """
        if not documents:
            return ""
        
        # If query is provided, rank documents by relevance
        if query:
            ranked_docs = self._rank_documents_by_relevance(documents, query)
            selected_docs = ranked_docs[:max_documents]
        else:
            selected_docs = documents[:max_documents]
        
        # Format each document
        context_parts = []
        for doc in selected_docs:
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', '')
            page_info = f" (page {page})" if page else ''
            
            # Format based on document type if available
            context_parts.append(f"From {source}{page_info}:\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _format_image_documents(self, 
                            documents: List[Document],
                            query: str = "",
                            max_documents: int = 3) -> str:
        """
        Format image documents with enhanced support for VLMs.
        
        Args:
            documents: List of Document objects
            query: Optional query for relevance-based formatting
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted image context
        """
        if not documents:
            return ""
        
        # Check if any documents appear to be for VLM processing
        has_vlm_hints = any("vlm" in str(doc.metadata).lower() for doc in documents)
        
        # Get formatted context and paths
        context, image_paths = self.get_image_context(
            documents, 
            max_images=max_documents,
            for_vlm=has_vlm_hints
        )
        
        if has_vlm_hints and image_paths:
            # Add special marker for VLM processing
            return f"{context}\n\n[NOTE: {len(image_paths)} images are available for vision processing]"
        
        return context
    
    def _format_audio_documents(self, 
                               documents: List[Document],
                               max_documents: int = 5) -> str:
        """
        Format audio document metadata.
        
        Args:
            documents: List of Document objects
            max_documents: Maximum number of documents to include
            
        Returns:
            Formatted audio context
        """
        if not documents:
            return ""
        
        selected_docs = documents[:max_documents]
        
        context_parts = []
        for doc in selected_docs:
            filename = os.path.basename(doc.metadata.get('source', 'Unknown audio'))
            file_type = doc.metadata.get('file_type', 'unknown')
            
            context_parts.append(
                f"Audio file: {filename} (Format: {file_type})"
            )
        
        return "\n".join(context_parts)
    
    def _rank_documents_by_relevance(self, 
                               documents: List[Document],
                               query: str) -> List[Document]:
        """
        Rank documents by relevance to a query using simple keyword matching.
        
        Args:
            documents: List of Document objects
            query: User query for relevance calculation
            
        Returns:
            List of documents sorted by relevance
        """
        if not query or not documents:
            return documents
            
        # Simple relevance calculation using keyword matching
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content = doc.page_content.lower()
            # Count matching words as a simple relevance score
            relevance_score = sum(1 for word in query_words if word in content)
            scored_docs.append((relevance_score, doc))
        
        # Sort by score in descending order
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Return just the documents, without scores
        return [doc for score, doc in scored_docs]

class RAGContentProcessor(BaseContentProcessor):
    """
    Process content specifically for RAG (Retrieval Augmented Generation).
    This processor focuses on preparing documents for vector storage and retrieval.
    """
    
    async def process_documents(self, 
                               documents: List[Document], 
                               **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process documents for RAG by preparing them for vector storage.
        
        Args:
            documents: List of Document objects
            kwargs: Additional processing parameters
                - embedding_model: Name of the embedding model to use
                - vector_store_type: Type of vector store (e.g., "faiss", "chroma")
                
        Returns:
            Tuple of (status message, processed data for vector store)
        """
        if not documents:
            return "No documents to process", {"documents": [], "metadata": {}}
        
        # Extract metadata for monitoring and tracking
        metadata = self.get_metadata_summary(documents)
        
        # Transform documents for RAG if needed
        # (e.g., standardize metadata, ensure IDs, etc.)
        processed_docs = self._prepare_documents_for_vectorstore(documents)
        
        # Generate a message about processed files
        files_message = self.format_files_message(documents)
        
        return (
            f"Processed {len(documents)} documents for RAG",
            {
                "documents": processed_docs, 
                "metadata": metadata,
                "files_message": files_message
            }
        )
    
    def _prepare_documents_for_vectorstore(self, 
                                         documents: List[Document]) -> List[Document]:
        """
        Prepare documents for insertion into a vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document objects ready for vectorization
        """
        processed_docs = []
        
        for idx, doc in enumerate(documents):
            # Ensure each document has a unique ID
            metadata = dict(doc.metadata)
            if "id" not in metadata:
                metadata["id"] = f"doc_{idx}"
                
            # Standardize document structure
            processed_docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        
        return processed_docs
    
    def get_retrieval_query(self, 
                          query: str, 
                          **kwargs) -> str:
        """
        Prepare a user query for retrieval from vector store.
        
        Args:
            query: User query
            kwargs: Additional parameters
                - enhance_query: Whether to enhance the query
                - max_query_length: Maximum query length
                
        Returns:
            Processed query for retrieval
        """
        enhance_query = kwargs.get("enhance_query", False)
        max_length = kwargs.get("max_query_length", 512)
        
        # Basic query processing
        processed_query = query.strip()
        
        # Truncate if needed
        if len(processed_query) > max_length:
            processed_query = processed_query[:max_length]
        
        # Future: Could add query enhancement logic here
        # (e.g., query expansion, entity extraction, etc.)
        
        return processed_query
    
    def format_retrieved_documents(self, 
                                 documents: List[Document], 
                                 **kwargs) -> str:
        """
        Format retrieved documents into a context string for LLM prompting.
        
        Args:
            documents: List of retrieved Document objects
            kwargs: Additional formatting parameters
                - separators: Custom separators for documents
                - include_metadata: Whether to include metadata
                
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        doc_separator = kwargs.get("doc_separator", "\n\n")
        include_metadata = kwargs.get("include_metadata", True)
        
        context_parts = []
        
        for idx, doc in enumerate(documents):
            # Format each document with optional metadata
            if include_metadata:
                source = doc.metadata.get("source", "Unknown source")
                doc_id = doc.metadata.get("id", f"document_{idx}")
                
                context_parts.append(
                    f"[Document {idx+1}] From {source} (ID: {doc_id}):\n{doc.page_content}"
                )
            else:
                context_parts.append(doc.page_content)
        
        return doc_separator.join(context_parts)

class SummarizationContentProcessor(BaseContentProcessor):
    """
    Process content specifically for summarization.
    """
    
    async def process_documents(self, 
                               documents: List[Document], 
                               **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process documents for summarization.
        
        Args:
            documents: List of Document objects
            kwargs: Additional processing parameters
                - method: Summarization method (e.g., "map_reduce", "refine", "stuff")
                - max_input_size: Maximum total input size to summarize
                
        Returns:
            Tuple of (status message, processed data for summarization)
        """
        if not documents:
            return "No documents to process", {"documents": [], "metadata": {}}
        
        # Extract parameters
        method = kwargs.get("method", "map_reduce")
        max_input_size = kwargs.get("max_input_size", 100000)
        
        # Get document metadata
        metadata = self.get_metadata_summary(documents)
        
        # Prepare documents based on summarization method
        prepared_docs = self._prepare_documents_for_method(
            documents, method, max_input_size)
        
        return (
            f"Processed {len(documents)} documents for summarization using {method} method",
            {"documents": prepared_docs, "metadata": metadata}
        )
    
    def _prepare_documents_for_method(self,
                                    documents: List[Document],
                                    method: str,
                                    max_input_size: int) -> List[Document]:
        """
        Prepare documents based on the summarization method.
        
        Args:
            documents: List of Document objects
            method: Summarization method
            max_input_size: Maximum total input size
            
        Returns:
            List of prepared Document objects
        """
        if method == "stuff":
            # For stuff method, we need to ensure the total content isn't too large
            return self._prepare_for_stuff(documents, max_input_size)
        elif method == "map_reduce":
            # For map-reduce, we might want to group documents by source
            return self._prepare_for_map_reduce(documents)
        elif method == "refine":
            # For refine, we might want to sort documents by relevance or date
            return self._prepare_for_refine(documents)
        else:
            # Default processing
            return documents
    
    def _prepare_for_stuff(self, 
                         documents: List[Document],
                         max_input_size: int) -> List[Document]:
        """
        Prepare documents for stuff summarization method.
        
        Args:
            documents: List of Document objects
            max_input_size: Maximum total input size
            
        Returns:
            List of prepared Document objects
        """
        # Check total content size
        total_size = sum(len(doc.page_content) for doc in documents)
        
        if total_size <= max_input_size:
            # If under size limit, return all documents
            return documents
        
        # If over limit, truncate or select most important documents
        # Simple strategy: just take documents until we hit the limit
        prepared_docs = []
        current_size = 0
        
        for doc in documents:
            doc_size = len(doc.page_content)
            if current_size + doc_size <= max_input_size:
                prepared_docs.append(doc)
                current_size += doc_size
            else:
                # If adding this document would exceed the limit, we're done
                break
        
        return prepared_docs
    
    def _prepare_for_map_reduce(self, documents: List[Document]) -> List[Document]:
        """
        Prepare documents for map-reduce summarization method.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of prepared Document objects
        """
        # Group documents by source for more coherent mapping
        docs_by_source = {}
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # Flatten grouped documents
        prepared_docs = []
        for source, docs in docs_by_source.items():
            # Sort by page number if available
            sorted_docs = sorted(
                docs, 
                key=lambda d: d.metadata.get("page", 0)
            )
            prepared_docs.extend(sorted_docs)
        
        return prepared_docs
    
    def _prepare_for_refine(self, documents: List[Document]) -> List[Document]:
        """
        Prepare documents for refine summarization method.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of prepared Document objects
        """
        # For refine method, the order matters
        # Sort by source and page number
        return sorted(
            documents,
            key=lambda d: (
                d.metadata.get("source", ""),
                d.metadata.get("page", 0)
            )
        )

class TranscriptionContentProcessor(BaseContentProcessor):
    """
    Process content specifically for transcription.
    Focuses on preparing audio files for transcription models.
    """
    
    async def process_documents(self, 
                               documents: List[Document], 
                               **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process documents for transcription.
        
        Args:
            documents: List of Document objects
            kwargs: Additional processing parameters
                - model_size: Size of the transcription model
                - language: Language of the audio
                - task_type: Task type (transcribe or translate)
                
        Returns:
            Tuple of (status message, processed data for transcription)
        """
        if not documents:
            return "No documents to process", {"audio_files": [], "metadata": {}}
        
        # Filter for audio documents
        audio_docs = [doc for doc in documents 
                     if doc.metadata.get("content_type") == "audio"]
        
        if not audio_docs:
            return "No audio documents found", {"audio_files": [], "metadata": {}}
        
        # Get document metadata
        metadata = self.get_metadata_summary(audio_docs)
        
        # Prepare audio files for transcription
        audio_files = self._extract_audio_files(audio_docs)
        
        return (
            f"Processed {len(audio_docs)} audio files for transcription",
            {"audio_files": audio_files, "metadata": metadata}
        )
    
    def _extract_audio_files(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract audio file information from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of dictionaries with audio file information
        """
        audio_files = []
        
        for doc in documents:
            # Get the processed audio path from metadata
            processed_path = doc.metadata.get("processed_path")
            if not processed_path:
                continue
                
            # Check if the file exists
            if not os.path.exists(processed_path):
                logger.warning(f"Processed audio file not found: {processed_path}")
                continue
                
            # Extract relevant metadata
            file_info = {
                "path": processed_path,
                "original_path": doc.metadata.get("original_path", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "sample_rate": doc.metadata.get("sample_rate", 16000)
            }
            
            audio_files.append(file_info)
        
        return audio_files