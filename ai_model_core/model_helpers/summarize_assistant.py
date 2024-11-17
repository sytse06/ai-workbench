# model_helpers/summarize_assistant.py
# Standard library imports
import logging
from typing import (
    List, Literal, TypedDict, Annotated, Union, Dict,
    Any, TypeVar, NotRequired
)
import operator
from dataclasses import dataclass
from enum import Enum, auto

# Third-party imports
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langgraph.constants import Send
from langchain_core.output_parsers import StrOutputParser

# Local imports
from ..shared_utils import (
    get_model,
    get_prompt_template,
)
from ..config.settings import load_config
from ..shared_utils.utils import EnhancedContentLoader

logger = logging.getLogger(__name__)

Method = Literal["stuff", "map_reduce", "refine"]
T = TypeVar('T', bound='BaseState')


class ErrorType(Enum):
    """Types of errors that can occur during summarization"""
    VALIDATION = auto()
    STATE = auto()
    MODEL = auto()
    GRAPH = auto()
    UNKNOWN = auto()


@dataclass
class SummarizationError(Exception):
    """Custom error for summarization failures"""
    error_type: ErrorType
    message: str
    state: Dict[str, Any]

    def __str__(self) -> str:
        return f"{self.error_type.name}: {self.message}"


class BaseState(TypedDict):
    """Base state interface for all operations"""
    chunks: List[Document]
    method: Method
    final_summary: str
    error: NotRequired[str]


class StuffState(BaseState):
    """State for stuff method"""
    pass


class MapReduceState(BaseState):
    """State for map-reduce method"""
    intermediate_summaries: Annotated[List[str], operator.add]


class RefineState(BaseState):
    """State for refine method"""
    content: str
    existing_summary: str


class GraphState(TypedDict):
    """Complete graph state with all possible fields"""
    chunks: List[Document]
    method: Method
    final_summary: str
    error: NotRequired[str]
    intermediate_summaries: NotRequired[List[str]]
    content: NotRequired[str]
    existing_summary: NotRequired[str]


def create_initial_state() -> GraphState:
    """Create initial state for the graph"""
    return {
        "chunks": [],
        "method": "stuff",
        "final_summary": "",
    }


class SummarizationAssistant:
    def __init__(
        self,
        model_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.4,
        method: str = "map_reduce",
        max_tokens: int = 1000,
        prompt_info: str = "summarize",
        language_choice: str = "english",
        verbose: bool = False
    ):
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.method = method
        self.temperature = temperature
        self.content_loader = EnhancedContentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.config = load_config()
        self.prompt_info = prompt_info
        self.language_choice = language_choice
        self.verbose = verbose
        
        msg = (
            "Initializing SummarizationAssistant with model: "
            f"{model_name}"
        )
        self.log_verbose(msg)
        
        # Load prompt templates
        prompt_types = [
            ("stuff_prompt", f"{self.prompt_info}_stuff"),
            ("map_prompt", f"{self.prompt_info}_map"),
            ("map_reduce_prompt", f"{self.prompt_info}_map_reduce"),
            ("reduce_prompt", "reduce_template"),
            ("refine_prompt", f"{self.prompt_info}_refine")
        ]
        
        # Load each prompt template
        config = (self.config, self.language_choice)
        for attr, template in prompt_types:
            setattr(self, attr, get_prompt_template(template, *config))
        
        self.log_verbose("Prompt templates loaded successfully")
        
        # Initialize graph and runnable as None
        self.graph = None
        self.graph_runnable = None

        self.valid_methods = ["stuff", "map_reduce", "refine"]
        if method not in self.valid_methods:
            raise SummarizationError(
                ErrorType.VALIDATION,
                f"Invalid method. Must be one of {self.valid_methods}",
                {"method": method}
            )
        
    def validate_state(self, state: GraphState) -> None:
        """Validate the graph state"""
        try:
            required = ["chunks", "method", "final_summary"]
            for key in required:
                if key not in state:
                    raise KeyError(f"Missing required key: {key}")
            
            if state["method"] not in self.valid_methods:
                raise ValueError(
                    f"Invalid method. Must be one of {self.valid_methods}"
                )
            
            if not isinstance(state["chunks"], list):
                raise TypeError("Chunks must be a list")

            # Method-specific validation
            if state["method"] == "map_reduce":
                if "intermediate_summaries" not in state:
                    raise KeyError(
                        "intermediate_summaries required for map_reduce"
                    )
                if not isinstance(state["intermediate_summaries"], list):
                    raise TypeError(
                        "intermediate_summaries must be a list"
                    )

            if state["method"] == "refine":
                if "content" not in state:
                    raise KeyError("content required for refine")
                if "existing_summary" not in state:
                    raise KeyError("existing_summary required for refine")

        except Exception as e:
            raise SummarizationError(
                ErrorType.VALIDATION,
                str(e),
                state
            ) from e

    def log_verbose(self, message: str):
        if self.verbose:
            logger.info(message)
            
    async def summarize_stuff(self, state: StuffState) -> Dict[str, str]:
        try:
            self.validate_state(state)
            self.log_verbose("Starting 'stuff' summarization method")
            
            if isinstance(state["chunks"][0], Document):
                texts = [chunk.page_content for chunk in state["chunks"]]
                combined_text = "\n\n".join(texts)
            else:
                combined_text = "\n\n".join(state["chunks"])
            
            self.log_verbose(
                f"Combined text length: {len(combined_text)} characters"
            )
            
            chain = self.stuff_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({
                "user_message": combined_text,   
                "prompt_info": self.prompt_info
            })
            
            self.log_verbose(
                f"'Stuff' summarization completed. Length: {len(summary)}"
            )
            return {"final_summary": summary}
        except Exception as e:
            raise SummarizationError(
                ErrorType.MODEL,
                str(e),
                state
            ) from e
    
    async def generate_map_summary(
        self, state: MapReduceState
    ) -> Dict[str, List[str]]:
        try:
            if "content" not in state:
                raise KeyError("State must contain 'content' key")
            if state["content"] is None:
                raise ValueError("content cannot be None")
                
            content_len = len(state['content'])
            self.log_verbose(
                f"Generating map summary for chunk length: {content_len}"
            )
            
            chain = self.map_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({"text": state["content"]})
            self.log_verbose(
                f"Map summary generated. Length: {len(summary)}"
            )
            return {"intermediate_summaries": [summary]}
        except Exception as e:
            raise SummarizationError(
                ErrorType.MODEL,
                str(e),
                state
            ) from e

    async def reduce_summaries(
        self, state: MapReduceState
    ) -> Dict[str, str]:
        try:
            self.validate_state(state)
            if not state["intermediate_summaries"]:
                raise ValueError("No intermediate summaries to reduce")
            
            count = len(state["intermediate_summaries"])
            self.log_verbose(f"Reducing {count} intermediate summaries")
            
            combined = "\n\n".join(state["intermediate_summaries"])
            chain = self.map_reduce_prompt | self.model | StrOutputParser()
            summary = await chain.ainvoke({"text": combined})
            
            self.log_verbose(f"Reduction completed. Length: {len(summary)}")
            return {"final_summary": summary}
        except Exception as e:
            raise SummarizationError(
                ErrorType.MODEL,
                str(e),
                state
            ) from e

    async def refine_summary(
        self, state: RefineState
    ) -> Dict[str, str]:
        try:
            self.validate_state(state)
            
            content_len = len(state["content"])
            self.log_verbose(
                f"Refining summary. New content length: {content_len}"
            )
            
            chain = self.refine_prompt | self.model | StrOutputParser()
            refined = await chain.ainvoke({
                "text": (
                    f"New content: {state['content']}\n\n"
                    f"Existing summary: {state['existing_summary']}"
                )
            })
            
            self.log_verbose(f"Summary refined. Length: {len(refined)}")
            return {"final_summary": refined}
        except Exception as e:
            raise SummarizationError(
                ErrorType.MODEL,
                str(e),
                state
            ) from e

    def map_summaries(
        self, state: MapReduceState
    ) -> List[Send]:
        try:
            method = state['method']
            self.log_verbose(f"Mapping summaries using method: {method}")
            if not state.get("chunks"):
                raise ValueError("No chunks to process")
                
            if state["method"] == "map_reduce":
                return [
                    Send(
                        "generate_map_summary",
                        {"content": chunk.page_content}
                    )
                    for chunk in state["chunks"]
                ]
            return []
        except Exception as e:
            raise SummarizationError(
                ErrorType.STATE,
                str(e),
                state
            ) from e

    def handle_error(
        self, error: Exception, state: GraphState
    ) -> GraphState:
        """Handle errors during graph execution"""
        if isinstance(error, SummarizationError):
            error_type = error.error_type
            message = error.message
        else:
            error_type = ErrorType.UNKNOWN
            message = str(error)

        self.log_verbose(f"Error occurred: {error_type.name} - {message}")
        
        # Return a valid state with error information
        return {
            "chunks": state.get("chunks", []),
            "method": state.get("method", self.method),
            "final_summary": f"Error: {message}",
            "error": f"{error_type.name}: {message}"
        }

    def setup_graph(self):
        """Set up the graph for summarization"""
        if self.graph is not None:
            return

        try:
            self.log_verbose("Setting up summarization graph")
            self.graph = StateGraph(create_initial_state)

            # Define the router function
            def router(state: GraphState) -> Dict[str, Any]:
                self.validate_state(state)
                method = state["method"]
                return {"next_step": {
                    "stuff": "summarize_stuff",
                    "map_reduce": "generate_map_summary",
                    "refine": "refine_summary"
                }[method], "method": method}

            # Define the decision function
            def decide_next_step(state: GraphState) -> str:
                return state["next_step"]

            # Add nodes
            nodes = {
                "router": router,
                "summarize_stuff": self.summarize_stuff,
                "generate_map_summary": self.generate_map_summary,
                "reduce_summaries": self.reduce_summaries,
                "refine_summary": self.refine_summary
            }
            for name, func in nodes.items():
                self.graph.add_node(name, func)

            # Add main edges
            self.graph.add_edge(START, "router")
            self.graph.add_conditional_edges(
                "router",
                decide_next_step,
                {
                    "stuff": "summarize_stuff",
                    "map_reduce": "generate_map_summary",
                    "refine": "refine_summary",
                }
            )
            
            # Add edges
            self.graph.add_edge("generate_map_summary", "reduce_summaries")
            self.graph.add_edge("summarize_stuff", END)
            self.graph.add_edge("reduce_summaries", END)
            self.graph.add_edge("refine_summary", END)

            # Add error handling
            self.graph.set_error_handler(self.handle_error)

            # Compile the graph
            self.graph_runnable = self.graph.compile()
            self.log_verbose("Summarization graph setup completed")
        except Exception as e:
            raise SummarizationError(
                ErrorType.GRAPH,
                str(e),
                {}
            ) from e
            
    async def summarize(
        self,
        content: Union[str, List[str], List[Document]],
        method: str = None,
        prompt_info: str = None,
        language: str = "english"
    ) -> Dict[str, Any]:
        try:
            method = method or self.method
            if method not in self.valid_methods:
                raise ValueError(
                    f"Invalid method. Must be one of {self.valid_methods}"
                )
                
            self.log_verbose(
                f"Starting summarization using {method} method"
            )
            self.log_verbose("Loading and splitting documents")

            if isinstance(content, str):
                # Assume it's a file path or URL
                chunks = self.content_loader.load_and_split_document(
                    content
                )
            elif isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    # List of file paths or URLs
                    chunks = self.content_loader.load_and_split_document(
                        content
                    )
                elif all(isinstance(item, Document) for item in content):
                    # List of Documents
                    chunks = self.content_loader.split_documents(
                        content,
                        self.chunk_size,
                        self.chunk_overlap
                    )
                else:
                    raise ValueError(
                        "Content list must be all strings or all Documents"
                    )
            else:
                raise ValueError(
                    "Content must be a string, list of strings, or Documents"
                )

            self.log_verbose(f"Processing {len(chunks)} chunks")

            # Set up graph if not already done
            if self.graph_runnable is None:
                self.setup_graph()

            # Create initial state based on method
            state: GraphState = {
                "chunks": chunks,
                "method": method,
                "final_summary": "",
            }
            if method == "map_reduce":
                state["intermediate_summaries"] = []
            elif method == "refine":
                state["content"] = ""
                state["existing_summary"] = ""

            result = await self.graph_runnable.ainvoke(state)
            
            summary_len = len(result['final_summary'])
            self.log_verbose(f"Summarization done. Length: {summary_len}")
            return result
        except Exception as e:
            if isinstance(e, SummarizationError):
                raise e
            raise SummarizationError(
                ErrorType.UNKNOWN,
                str(e),
                locals().get("state", {})
            ) from e
