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
    def __init__(self, model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, temperature: float = 0.4, method: str = "map_reduce", max_tokens: int = 1000, prompt_info: str = "summarize", language_choice: str = "english", verbose: bool = False):
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
        """Load all required prompt templates."""
        try:
            self.stuff_prompt = get_prompt_template(
                f"{self.prompt_info}_stuff", 
                self.config, 
                self.language_choice
            )
            self.map_prompt = get_prompt_template(
                f"{self.prompt_info}_map", 
                self.config, 
                self.language_choice
            )
            self.map_reduce_prompt = get_prompt_template(
                f"{self.prompt_info}_map_reduce", 
                self.config, 
                self.language_choice
            )
            self.reduce_prompt = get_prompt_template(
                "reduce_template", 
                self.config, 
                self.language_choice
            )
            self.refine_prompt = get_prompt_template(
                f"{self.prompt_info}_refine", 
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
        try:
            self.log_verbose("Starting 'stuff' summarization method")
            
            # Handle both Document objects and strings
            texts = []
            for chunk in state["chunks"]:
                if isinstance(chunk, Document):
                    texts.append(chunk.page_content)
                else:
                    texts.append(chunk)
            
            combined_text = "\n\n".join(texts)
            
            if not combined_text.strip():
                raise SummarizeException("No content to summarize")
                
            chain = self.stuff_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "user_message": combined_text
            })
            
            # Validate and truncate if necessary
            if len(summary) > self.max_tokens:
                summary = summary[:self.max_tokens]
            
            self.log_verbose(f"'Stuff' summarization completed. Summary length: {len(summary)}")
            return {"final_summary": summary}
            
        except Exception as e:
            logger.error(f"Error in stuff method: {str(e)}")
            raise SummarizeException(f"Stuff method failed: {str(e)}")
    
    async def generate_map_summary(self, state: SummaryState) -> dict:
        """Generate summary for an individual chunk in the map phase."""
        try:
            chunk = state["chunk"]
            if isinstance(chunk, Document):
                text = chunk.page_content
            else:
                text = chunk
                
            if not text.strip():
                raise SummarizeException("Empty chunk received")
                
            self.log_verbose(f"Generating map summary for chunk of length: {len(text)}")
            
            chain = self.map_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "user_message": text
            })
            
            return {"intermediate_summaries": [summary]}
            
        except Exception as e:
            logger.error(f"Error in map phase: {str(e)}")
            raise SummarizeException(f"Map phase failed: {str(e)}")

    async def reduce_summaries(self, state: OverallState) -> dict:
        """Combine multiple summaries into a final summary."""
        try:
            self.log_verbose(f"Reducing {len(state['intermediate_summaries'])} summaries")
            
            if not state['intermediate_summaries']:
                raise SummarizeException("No intermediate summaries to reduce")
                
            combined_summaries = "\n\n".join(state['intermediate_summaries'])
            
            chain = self.reduce_prompt | self.model | StrOutputParser()
            final_summary = await chain.ainvoke({
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "user_message": combined_summaries
            })
            
            if len(final_summary) > self.max_tokens:
                final_summary = final_summary[:self.max_tokens]
            
            return {"final_summary": final_summary}
            
        except Exception as e:
            logger.error(f"Error in reduce phase: {str(e)}")
            raise SummarizeException(f"Reduce phase failed: {str(e)}")

    async def refine_summary(self, state: SummaryState) -> dict:
        """Refine an existing summary with new content."""
        try:
            chunk = state["chunk"]
            if isinstance(chunk, Document):
                text = chunk.page_content
            else:
                text = chunk
                
            if not text.strip():
                raise SummarizeException("Empty chunk received for refinement")
                
            self.log_verbose("Refining summary with new content")
            
            chain = self.refine_prompt | self.model | StrOutputParser()
            refined_summary = await chain.ainvoke({
                "prompt_info": state.get("prompt_info", self.prompt_info),
                "user_message": f"existing_summary: {state.get('existing_summary', '')}\ntext: {text}"
            })
            
            if len(refined_summary) > self.max_tokens:
                refined_summary = refined_summary[:self.max_tokens]
            
            return {"final_summary": refined_summary}
            
        except Exception as e:
            logger.error(f"Error in refine phase: {str(e)}")
            raise SummarizeException(f"Refine phase failed: {str(e)}")

    def map_summaries(self, state: OverallState) -> List[Send]:
        """Create a list of Send objects for processing chunks based on method."""
        try:
            self.log_verbose(f"Mapping summaries using {state['method']} method")
            
            if not state["chunks"]:
                raise SummarizeException("No chunks to process")
                
            if state["method"] == "map_reduce":
                return [
                    Send("generate_map_summary", {
                        "chunk": chunk,
                        "prompt_info": state.get("prompt_info", self.prompt_info)
                    })
                    for chunk in state["chunks"]
                ]
            elif state["method"] == "refine":
                return [
                    Send("refine_summary", {
                        "chunk": chunk,
                        "existing_summary": state.get("final_summary", ""),
                        "prompt_info": state.get("prompt_info", self.prompt_info)
                    })
                    for chunk in state["chunks"]
                ]
            else:  # stuff method
                return [Send("summarize_stuff", state)]
                
        except Exception as e:
            logger.error(f"Error in mapping phase: {str(e)}")
            raise SummarizeException(f"Mapping phase failed: {str(e)}")

    def setup_graph(self):
        """Set up the langgraph workflow for all summarization methods."""
        try:
            self.log_verbose("Setting up summarization graph")
            
            def router(state: OverallState) -> dict:
                """Route to appropriate processing method."""
                method = state["method"]
                if method == "stuff":
                    return {"next_step": "summarize_stuff"}
                elif method in ["map_reduce", "refine"]:
                    return {"next_step": "map_summaries"}
                else:
                    raise ValueError(f"Unknown method: {method}")

            def decide_next_step(state: OverallState) -> str:
                """Determine the next step in the workflow."""
                method = state["method"]
                if method == "map_reduce" and state.get("intermediate_summaries"):
                    return "reduce_summaries"
                return state.get("next_step", END)

            # Add nodes
            self.graph.add_node("router", router)
            self.graph.add_node("summarize_stuff", self.summarize_stuff)
            self.graph.add_node("map_summaries", self.map_summaries)
            self.graph.add_node("generate_map_summary", self.generate_map_summary)
            self.graph.add_node("reduce_summaries", self.reduce_summaries)
            self.graph.add_node("refine_summary", self.refine_summary)

            # Add edges
            self.graph.add_edge(START, "router")
            self.graph.add_conditional_edges(
                "router",
                decide_next_step,
                {
                    "summarize_stuff": "summarize_stuff",
                    "map_summaries": "map_summaries",
                    "generate_map_summary": "generate_map_summary",
                    "reduce_summaries": "reduce_summaries",
                    "refine_summary": "refine_summary"
                }
            )
            
            # Add method-specific edges
            self.graph.add_edge("map_summaries", "generate_map_summary")
            self.graph.add_edge("map_summaries", "refine_summary")
            self.graph.add_edge("generate_map_summary", "reduce_summaries")
            self.graph.add_edge("reduce_summaries", END)
            self.graph.add_edge("refine_summary", END)
            self.graph.add_edge("summarize_stuff", END)

            # Compile the graph
            self.graph_runnable = self.graph.compile()
            self.log_verbose("Summarization graph setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise

    async def summarize(self, chunks: Union[str, List[str], List[Document]], method: str = None, prompt_info: str = None, language: str = "english") -> dict:
        """Main entry point for summarization."""
        try:
            self.log_verbose(f"Starting summarization process using {method} method")
            self.log_verbose(f"Loading and splitting documents")

            if isinstance(chunks, str):
                # Assume it's a file path or URL
                chunks = self.content_loader.load_and_split_documents(file_paths=chunks)
            elif isinstance(chunks, list):
                if all(isinstance(item, str) for item in chunks):
                    # List of file paths or URLs
                    chunks = self.content_loader.load_and_split_documents(file_paths=chunks)
                elif all(isinstance(item, Document) for item in chunks):
                    # List of Documents
                    chunks = self.content_loader.split_documents(chunks, self.chunk_size, self.chunk_overlap)
                else:
                    raise ValueError("Content list must contain either all strings (file paths/URLs) or all Documents")
            else:
                raise ValueError("Content must be either a string, a list of strings, or a list of Documents")

            self.log_verbose(f"Processing {len(chunks)} chunks")

            if not self.graph_runnable:
                self.setup_graph()

            result = await self.graph_runnable.ainvoke({
                "chunks": chunks,
                "method": method or self.method,
                "language": language,
                "intermediate_summaries": [],
                "final_summary": "",
                "prompt_info": prompt_info or self.prompt_info
            })
            self.log_verbose(f"Summarization completed. Final summary length: {len(result['final_summary'])} characters")
            return result
                
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            raise SummarizeException(f"Summarization failed: {str(e)}")