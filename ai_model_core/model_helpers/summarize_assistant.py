# model_helpers/summarize_assistant.py
# Standard library imports
import logging
from typing import List, Literal, TypedDict, Annotated, Union, Optional
import operator
from operator import add

# Third-party imports
from langgraph.graph import StateGraph, END, START
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
    intermediate_summaries: Annotated[List[str], operator.add]
    final_summary: str
    method: Literal["stuff", "map_reduce", "refine"]
    prompt_info: str
    language: str
    next_step: Optional[str]

class SummaryState(TypedDict):
    """State for individual summary operations."""
    chunk: Union[str, Document]
    existing_summary: str
    prompt_info: str
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
            
    async def summarize_stuff(self, state: OverallState) -> dict:
        """Summarize all content at once using the stuff method."""
        self.log_verbose("Starting 'stuff' summarization method")
        
        texts = [chunk.page_content if isinstance(chunk, Document) else chunk 
                for chunk in state["chunks"]]
        combined_text = "\n\n".join(texts)
        
        chain = self.stuff_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "user_message": combined_text,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        if len(summary) > self.max_tokens:
            summary = summary[:self.max_tokens]
        
        return {
            "final_summary": summary,
            "next_step": END  # Update state with next step
        }
            
    async def generate_map_summary(self, state: dict) -> dict:
        """Generate initial summary for a chunk in the map phase."""
        chunk = state["chunk"]
        text = chunk.page_content if isinstance(chunk, Document) else chunk
        
        chain = self.map_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "user_message": text,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        return {
            "intermediate_summaries": [summary],
            "next_step": "reduce_summaries"
        }

    async def reduce_summaries(self, state: OverallState) -> dict:
        """Combine multiple summaries into a final summary."""
        combined_text = "\n\n".join(state['intermediate_summaries'])
        
        chain = self.reduce_prompt | self.model | StrOutputParser()
        final_summary = await chain.ainvoke({
            "user_message": combined_text,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        if len(final_summary) > self.max_tokens:
            final_summary = final_summary[:self.max_tokens]
        
        return {
            "final_summary": final_summary,
            "next_step": END
        }
    
    def route_by_method(self, state: OverallState) -> dict:
        """Route to appropriate processing method based on state."""
        method = state["method"]
            
        if method == "stuff":
            return {"next_step": "summarize_stuff"}
        elif method == "map_reduce":
            return {"next_step": "map_summaries"}
        else:
            raise ValueError(f"Unsupported method: {method}")

    def map_summaries(self, state: OverallState) -> dict:
        """Create parallel processing tasks for chunks."""
        try:
            self.log_verbose(f"Mapping summaries using {state['method']} method")
            
            if not state["chunks"]:
                raise SummarizeException("No chunks to process")
            
            # Create Send objects for parallel processing
            sends = [
                Send("generate_map_summary", {
                    "chunk": chunk,
                    "prompt_info": state.get("prompt_info", self.prompt_info)
                })
                for chunk in state["chunks"]
            ]
            
            return {"sends": sends}
            
        except Exception as e:
            logger.error(f"Error in mapping phase: {str(e)}")
            raise SummarizeException(f"Mapping phase failed: {str(e)}")

    async def refine_summary(self, state: dict) -> dict:
        """Refine an existing summary with new content."""
        chunk = state["chunk"]
        existing_summary = state.get("existing_summary", "")
        text = chunk.page_content if isinstance(chunk, Document) else chunk
        
        if not existing_summary:
            chain = self.initial_refine_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "user_message": text,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            })
        else:
            chain = self.sequential_refine_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "user_message": text,
                "existing_summary": existing_summary,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            })
        
        if len(summary) > self.max_tokens:
            summary = summary[:self.max_tokens]
        
        return {
            "final_summary": summary,
            "next_step": END  # Update state to end
        }
            
    def setup_graph(self):
        """Set up the langgraph workflow."""
        try:
            self.log_verbose("Setting up summarization graph")            
            
            # Add nodes
            self.graph.add_node("router", self.route_by_method)
            self.graph.add_node("summarize_stuff", self.summarize_stuff)
            self.graph.add_node("map_summaries", self.map_summaries)
            self.graph.add_node("generate_map_summary", self.generate_map_summary)
            self.graph.add_node("reduce_summaries", self.reduce_summaries)

            # Set up edges for stuff method
            self.graph.add_edge(START, "router")
            self.graph.add_edge("router", "summarize_stuff")
            self.graph.add_edge("summarize_stuff", END)

            # Set up edges for map-reduce method with parallel processing
            self.graph.add_edge("router", "map_summaries")
            # This enables parallel processing of chunks
            self.graph.add_conditional_edges(
                "map_summaries",
                lambda x: {"sends": x["sends"]},
                ["generate_map_summary"]
            )
            self.graph.add_edge("generate_map_summary", "reduce_summaries")
            self.graph.add_edge("reduce_summaries", END)

            self.graph_runnable = self.graph.compile()
            
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise
                            
    async def summarize(self, chunks: Union[str, List[str], List[Document]], 
                       method: str = None, prompt_info: str = None, 
                       language: str = None) -> dict:
        """Main entry point for summarization."""
        method = method or self.method
        self.log_verbose(f"Starting summarization using {method} method")
        
        # Process input chunks using EnhancedContentLoader
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
            "language": language or self.language_choice,
            "intermediate_summaries": [],  # Will be combined using operator.add
            "final_summary": "",
            "prompt_info": prompt_info or self.prompt_info
        }

        result = await self.graph_runnable.ainvoke(initial_state)
        
        if not result.get("final_summary"):
            raise SummarizeException("No summary generated")
        
        return result