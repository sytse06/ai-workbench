# model_helpers/summarize_assistant.py
# Standard library imports
import logging
import asyncio
from typing import List, Literal, TypedDict, Annotated, Union, Optional
import operator
from operator import add

# Third-party imports
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langchain_core.documents import Document
from langgraph.constants import Send
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ai_model_core import (
    get_model,
    get_embedding_model,
    get_prompt_template,
    _format_history
)
from ai_model_core.config.settings import load_config
from ai_model_core.shared_utils import EnhancedContentLoader

logger = logging.getLogger(__name__)

class OverallState(TypedDict):
    """State for the overall summarization process."""
    chunks: List[Document]
    intermediate_summaries: List[str]
    final_summary: str
    method: Literal["stuff", "map_reduce"]
    prompt_info: str
    language: str
    existing_summary: Optional[str]
    chunk_tasks: Optional[List]

class SummaryState(TypedDict):
    """State for individual summary operations."""
    chunk: Union[str, Document]
    existing_summary: str
    prompt_info: str
    language: str # For tracking parallel processing
class SummarizeException(Exception):
    """Custom exception for summarization errors."""
    pass
class SummarizationAssistant:
    def __init__(self, model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, temperature: float = 0.4, method: str = "map_reduce", max_tokens: int = 3000, prompt_info: str = "summarize", language_choice: str = "english", verbose: bool = False):
        """Initialize the summarization assistant."""
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.method = method
        self.temperature = temperature
        self.prompt_info = prompt_info
        self.language_choice = language_choice
        self.verbose = verbose
        
        self.log_verbose(f"Initializing SummarizationAssistant with model: {model_name}")
        
        # Load configuration
        try:
            self.config = load_config()
            self.content_loader = EnhancedContentLoader(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            # Load prompt templates
            self._load_prompt_templates()
            
            # Initialize graph
            self.graph = StateGraph(OverallState)
            self.graph_runnable = None
            self.setup_graph()
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
        
    def _load_prompt_templates(self):
        """Load all required prompt templates with standardized names."""
        try:
            # Stuff method template
            self.stuff_prompt = get_prompt_template(
                "summary_stuff", 
                self.config, 
                self.language_choice
            )
            
            # Map-reduce method templates
            self.map_prompt = get_prompt_template(
                "initial_map_reduce_summary", 
                self.config, 
                self.language_choice
            )
            self.reduce_prompt = get_prompt_template(
                "sequential_map_reduce_summary", 
                self.config, 
                self.language_choice
            )
            
            # Refine method templates
            self.initial_refine_prompt = get_prompt_template(
                "initial_refine_summary", 
                self.config, 
                self.language_choice
            )
            self.sequential_refine_prompt = get_prompt_template(
                "sequential_refine_summary", 
                self.config, 
                self.language_choice
            )
            
            self.log_verbose("Prompt templates loaded successfully")
        except Exception as e:
            logger.error(f"Error loading prompt templates: {str(e)}")
            raise

    def log_verbose(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)
    
    def route_by_method(self, state: OverallState) -> Command:
        """Route to appropriate method and prepare state accordingly."""
        method = state["method"]
        
        if method == "stuff":
            return Command(
                goto="summarize_stuff",
                update={}  # No special preparation needed
            )
        elif method == "map_reduce":
            # Prepare state for parallel processing
            return Command(
                goto="map_summaries",
                update={
                    "intermediate_summaries": [],  # Reset for new parallel processing
                    "chunk_tasks": []  # Prepare for parallel task tracking
                }
            )
        elif method == "refine":
            return Command(
                goto="refine_summary",
                update={"existing_summary": ""}  # Initialize refine state
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
                    
    async def summarize_stuff(self, state: OverallState) -> Command:
        """Summarize all content at once using stuff method."""
        try:
            self.log_verbose("Using stuff method for summarization")
            
            texts = [chunk.page_content if isinstance(chunk, Document) else chunk 
                    for chunk in state["chunks"]]
            combined_text = "\n\n".join(texts)
            
            chain = self.stuff_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "user_message": combined_text,
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "language": state.get("language", self.language_choice)
            })
            
            if len(summary) > self.max_tokens:
                summary = summary[:self.max_tokens]
            
            return Command(
                goto=END,
                update={"final_summary": summary}
            )
        except Exception as e:
            logger.error(f"Error in stuff method: {str(e)}")
            raise SummarizeException(f"Stuff method failed: {str(e)}")
                    
    async def process_single_chunk(self, chunk: Union[Document, str], 
                                 prompt_info: str, language: str) -> str:
        """Process a single chunk in the map phase."""
        text = chunk.page_content if isinstance(chunk, Document) else chunk
        
        chain = self.map_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "user_message": text,
            "prompt_info": prompt_info,
            "language": language
        })
        
        return summary

    async def map_summaries(self, state: OverallState) -> Command:
        """Process chunks in parallel for map phase."""
        try:
            self.log_verbose(f"Starting map phase with {len(state['chunks'])} chunks")
            
            if not state["chunks"]:
                raise SummarizeException("No chunks to process")

            # Create tasks for parallel processing
            tasks = [
                self.process_single_chunk(
                    chunk,
                    state.get("prompt_info", self.prompt_info),
                    state.get("language", self.language_choice)
                )
                for chunk in state["chunks"]
            ]
            
            # Execute all chunk processing in parallel
            summaries = await asyncio.gather(*tasks)
            
            self.log_verbose(f"Completed map phase, processed {len(summaries)} chunks")
            
            return Command(
                goto="reduce_summaries",
                update={"intermediate_summaries": summaries}
            )
                
        except Exception as e:
            logger.error(f"Error in map phase: {str(e)}")
            raise SummarizeException(f"Map phase failed: {str(e)}")

    async def reduce_summaries(self, state: OverallState) -> Command:
        """Combine summaries into final output."""
        try:
            self.log_verbose("Starting reduce phase")
            combined_text = "\n\n=== Section Summary ===\n".join(state['intermediate_summaries'])
            
            chain = self.reduce_prompt | self.model | StrOutputParser()
            final_summary = await chain.ainvoke({
                "user_message": combined_text,
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "language": state.get("language", self.language_choice)
            })
            
            if len(final_summary) > self.max_tokens:
                final_summary = final_summary[:self.max_tokens]
            
            return Command(
                goto=END,
                update={"final_summary": final_summary}
            )
            
        except Exception as e:
            logger.error(f"Error in reduce phase: {str(e)}")
            raise SummarizeException(f"Reduce phase failed: {str(e)}")
                
    async def refine_summary(self, state: OverallState) -> Command:
        """Refine summary iteratively with new content."""
        try:
            chunk = state["chunks"][0] if state["chunks"] else None
            if not chunk:
                return Command(
                    goto=END,
                    update={"final_summary": state.get("existing_summary", "")}
                )
            
            text = chunk.page_content if isinstance(chunk, Document) else chunk
            existing_summary = state.get("existing_summary", "")
            
            if not existing_summary:
                chain = self.initial_refine_prompt | self.model | StrOutputParser()
                new_summary = await chain.ainvoke({
                    "user_message": text,
                    "prompt_info": state.get("prompt_info", self.prompt_info),
                    "language": state.get("language", self.language_choice)
                })
            else:
                chain = self.sequential_refine_prompt | self.model | StrOutputParser()
                new_summary = await chain.ainvoke({
                    "user_message": text,
                    "existing_summary": existing_summary,
                    "prompt_info": state.get("prompt_info", self.prompt_info),
                    "language": state.get("language", self.language_choice)
                })
            
            remaining_chunks = state["chunks"][1:]
            
            if not remaining_chunks:
                return Command(
                    goto=END,
                    update={"final_summary": new_summary}
                )
            
            return Command(
                goto="refine_summary",
                update={
                    "chunks": remaining_chunks,
                    "existing_summary": new_summary
                }
            )
        except Exception as e:
            logger.error(f"Error in refine method: {str(e)}")
            raise SummarizeException(f"Refine method failed: {str(e)}")
            
    def setup_graph(self):
        """Set up graph with all summarization methods."""
        try:
            self.log_verbose("Setting up summarization graph")            
            
            # Add nodes for all methods
            self.graph.add_node("router", self.route_by_method)
            self.graph.add_node("summarize_stuff", self.summarize_stuff)
            self.graph.add_node("map_summaries", self.map_summaries)
            self.graph.add_node("reduce_summaries", self.reduce_summaries)
            self.graph.add_node("refine_summary", self.refine_summary)
            
            # Set entry point to router
            self.graph.set_entry_point("router")
            
            self.graph_runnable = self.graph.compile()
            
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise

    async def summarize(self, chunks: Union[str, List[str], List[Document]], 
                       method: str = None, prompt_info: str = None, 
                       language: str = None) -> dict:
        """Main entry point with support for all methods."""
        method = method or self.method
        language = language or self.language_choice
        
        self.log_verbose(f"Starting summarization using {method} method in {language}")
        
        # Process chunks using EnhancedContentLoader
        if isinstance(chunks, str):
            chunks = self.content_loader.load_and_split_documents(file_paths=chunks)
        elif isinstance(chunks, list):
            if all(isinstance(item, str) for item in chunks):
                chunks = self.content_loader.load_and_split_documents(file_paths=chunks)
            elif all(isinstance(item, Document) for item in chunks):
                chunks = self.content_loader.split_documents(chunks, self.chunk_size, self.chunk_overlap)
            else:
                raise ValueError("Content list must contain either all strings or all Documents")

        if not chunks:
            raise SummarizeException("No content to summarize after processing")

        initial_state = {
            "chunks": chunks,
            "method": method,
            "language": language,
            "intermediate_summaries": [],
            "final_summary": "",
            "existing_summary": "",  # For refine method
            "chunk_tasks": [],  # For parallel processing
            "prompt_info": prompt_info or self.prompt_info
        }

        result = await self.graph_runnable.ainvoke(initial_state)
        
        if not result.get("final_summary"):
            raise SummarizeException("No summary generated")
        
        return result