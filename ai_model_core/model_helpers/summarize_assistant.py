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
        try:
            self.log_verbose("Starting 'stuff' summarization method")
            logger.debug(f"Received state keys: {state.keys()}")
            
            if "chunks" not in state:
                logger.error("No chunks found in state")
                raise SummarizeException("No chunks in state")
                
            texts = []
            for i, chunk in enumerate(state["chunks"]):
                logger.debug(f"Processing chunk {i}")
                if isinstance(chunk, Document):
                    texts.append(chunk.page_content)
                else:
                    texts.append(chunk)
            
            logger.debug(f"Processed {len(texts)} chunks")
            combined_text = "\n\n".join(texts)
            
            if not combined_text.strip():
                logger.error("Combined text is empty")
                raise SummarizeException("No content to summarize")
                
            logger.debug(f"Combined text length: {len(combined_text)}")
            logger.debug("Creating chain with model and prompt")
            
            # Log the prompt template being used
            logger.debug(f"Using prompt template: {self.stuff_prompt}")
            
            chain = self.stuff_prompt | self.model | StrOutputParser()
            logger.debug("Invoking chain")
            
            summary = await chain.ainvoke({
                "user_message": combined_text,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            })
            
            logger.debug(f"Generated summary length: {len(summary) if summary else 0}")
            
            if not summary:
                logger.error("Chain returned empty summary")
                raise SummarizeException("Model returned empty summary")
                
            if len(summary) > self.max_tokens:
                logger.debug(f"Truncating summary from {len(summary)} to {self.max_tokens} tokens")
                summary = summary[:self.max_tokens]
            
            logger.debug("Successfully generated summary")
            return {"final_summary": summary}
                
        except Exception as e:
            logger.error(f"Error in stuff method: {str(e)}")
            if hasattr(e, "__traceback__"):
                logger.error("Full traceback:", exc_info=True)
            raise SummarizeException(f"Stuff method failed: {str(e)}")
            
    async def generate_map_summary(self, state: dict) -> dict:
        """Generate initial summary for a chunk in the map phase."""
        try:
            if not isinstance(state, dict):
                raise SummarizeException("Invalid state type")
            
            # Extract chunk from state or current_chunk
            chunk = state.get("chunk") or state.get("current_chunk")
            if chunk is None:
                raise SummarizeException("No chunk found in state")
                    
            text = chunk.page_content if isinstance(chunk, Document) else chunk
            if not text.strip():
                raise SummarizeException("Empty chunk received")
                    
            chain = self.map_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "user_message": text,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            })
            
            return {"intermediate_summaries": [summary]}
                
        except Exception as e:
            logger.error(f"Error in map phase: {str(e)}")
            raise SummarizeException(f"Map phase failed: {str(e)}")
    
    async def reduce_summaries(self, state: OverallState) -> dict:
        """Combine multiple summaries into a final summary."""
        try:
            if not state['intermediate_summaries']:
                raise SummarizeException("No intermediate summaries to reduce")
                
            combined_text = "\n\n".join(state['intermediate_summaries'])
            
            chain = self.reduce_prompt | self.model | StrOutputParser()
            final_summary = await chain.ainvoke({
                "user_message": combined_text,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            })
            
            if len(final_summary) > self.max_tokens:
                final_summary = final_summary[:self.max_tokens]
            
            return {"final_summary": final_summary}
            
        except Exception as e:
            logger.error(f"Error in reduce phase: {str(e)}")
            raise SummarizeException(f"Reduce phase failed: {str(e)}")

    async def refine_summary(self, state: dict) -> dict:
        """Refine an existing summary with new content."""
        try:
            if not isinstance(state, dict):
                raise SummarizeException("Invalid state type")
                
            chunk = state.get("chunk") or state.get("current_chunk")
            if chunk is None:
                raise SummarizeException("No chunk found in state")
                
            existing_summary = state.get("existing_summary", "")
            
            text = chunk.page_content if isinstance(chunk, Document) else chunk
                    
            if not text.strip():
                raise SummarizeException("Empty chunk received for refinement")
                    
            # Use initial refine prompt if no existing summary
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
            
            return {"final_summary": summary}
                
        except Exception as e:
            logger.error(f"Error in refine phase: {str(e)}")
            raise SummarizeException(f"Refine phase failed: {str(e)}")

    def map_summaries(self, state: OverallState) -> dict:
        """Create a list of Send objects for processing chunks based on method."""
        try:
            self.log_verbose(f"Mapping summaries using {state['method']} method")
            
            if not state["chunks"]:
                raise SummarizeException("No chunks to process")
                
            sends = []
            if state["method"] == "map_reduce":
                sends = [
                    Send("generate_map_summary", {
                        "chunk": chunk,
                        "prompt_info": state.get("prompt_info", self.prompt_info)
                    })
                    for chunk in state["chunks"]
                ]
            elif state["method"] == "refine":
                sends = [
                    Send("refine_summary", {
                        "chunk": chunk,
                        "existing_summary": state.get("final_summary", ""),
                        "prompt_info": state.get("prompt_info", self.prompt_info)
                    })
                    for chunk in state["chunks"]
                ]
            
            return {"sends": sends}
                
        except Exception as e:
            logger.error(f"Error in mapping phase: {str(e)}")
            raise SummarizeException(f"Mapping phase failed: {str(e)}")
            
    def setup_graph(self):
        """Set up the langgraph workflow."""
        try:
            self.log_verbose("Setting up summarization graph")
            
            def router(state: OverallState) -> dict:
                """Route to appropriate processing method."""
                method = state.get("method", self.method)
                logger.debug(f"Router called with method: {method}")
                
                # Return dict with next_step key to update state
                if method == "stuff":
                    return {"next_step": "summarize_stuff"}
                elif method == "map_reduce":
                    return {"next_step": "map_summaries"}
                elif method == "refine":
                    return {"next_step": "map_summaries"}
                else:
                    raise ValueError(f"Unknown method: {method}")

            def decide_next_step(state: OverallState) -> str:
                """Determine the next step in the workflow."""
                method = state.get("method", self.method)
                current_step = state.get("next_step")
                logger.debug(f"Deciding next step for method: {method}")
                
                if method == "stuff":
                    return END
                
                if method == "map_reduce":
                    if state.get("intermediate_summaries"):
                        logger.debug("Moving to reduce_summaries")
                        return "reduce_summaries"
                    logger.debug("Moving to generate_map_summary")
                    return "generate_map_summary"
                        
                if method == "refine":
                    logger.debug("Moving to refine_summary")
                    return "refine_summary"
                
                logger.debug("No matching condition found, ending")
                return END

            # Rest of the setup remains the same...
            self.graph = StateGraph(OverallState)
            
            # Add nodes
            self.graph.add_node("router", router)
            self.graph.add_node("summarize_stuff", self.summarize_stuff)
            self.graph.add_node("map_summaries", self.map_summaries)
            self.graph.add_node("generate_map_summary", self.generate_map_summary)
            self.graph.add_node("reduce_summaries", self.reduce_summaries)
            self.graph.add_node("refine_summary", self.refine_summary)

            # Add edges
            self.graph.add_edge(START, "router")
            self.graph.add_edge("summarize_stuff", END)
            self.graph.add_edge("map_summaries", "generate_map_summary")
            self.graph.add_edge("generate_map_summary", "reduce_summaries")
            self.graph.add_edge("reduce_summaries", END)
            self.graph.add_edge("map_summaries", "refine_summary")
            self.graph.add_edge("refine_summary", END)

            # Add conditional edges
            self.graph.add_conditional_edges(
                "router",
                decide_next_step,
                {
                    "summarize_stuff": "summarize_stuff",
                    "map_summaries": "map_summaries",
                    "generate_map_summary": "generate_map_summary",
                    "reduce_summaries": "reduce_summaries",
                    "refine_summary": "refine_summary",
                    END: END
                }
            )

            self.graph_runnable = self.graph.compile()
            self.log_verbose("Summarization graph setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up graph: {str(e)}")
            raise
    
    async def summarize(self, chunks: Union[str, List[str], List[Document]], 
                    method: str = None, prompt_info: str = None, 
                    language: str = None) -> dict:
        """Main entry point for summarization."""
        try:
            method = method or self.method
            language = language or self.language_choice
            self.log_verbose(f"Starting summarization process using {method} method")
            
            # Process input chunks
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

            self.log_verbose(f"Processing {len(chunks)} chunks")
            logger.debug(f"First chunk sample: {chunks[0].page_content[:100]}...")

            # Initialize state with all required fields, maintaining original prompt handling
            initial_state = {
                "chunks": chunks,
                "method": method,
                "language": language,
                "intermediate_summaries": [],
                "final_summary": "",
                "prompt_info": prompt_info or self.prompt_info,
                "next_step": None
            }

            logger.debug(f"Initial state: {initial_state}")
            if not self.graph_runnable:
                logger.error("Graph not initialized")
                raise SummarizeException("Graph not initialized")

            try:
                result = await self.graph_runnable.ainvoke(initial_state)
                logger.debug(f"Graph result: {result}")
            except Exception as graph_error:
                logger.error(f"Graph execution error: {str(graph_error)}")
                raise SummarizeException(f"Graph execution failed: {str(graph_error)}")
            
            if not result:
                logger.error("Empty result from graph")
                raise SummarizeException("Empty result from graph")
                
            if not result.get("final_summary"):
                logger.error(f"No final summary in result. Result keys: {result.keys()}")
                raise SummarizeException("No summary generated")
                
            self.log_verbose(f"Summarization completed. Summary length: {len(result['final_summary'])} characters")
            return result
                
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            raise SummarizeException(f"Summarization failed: {str(e)}")