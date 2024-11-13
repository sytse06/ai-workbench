# Standard library imports
import logging
from typing import List, Literal, TypedDict, Annotated, Union
import operator

# Third-party imports
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ..shared_utils import (
    get_model,
    get_prompt_template,
)
from ..config.settings import load_config
from ..shared_utils.utils import EnhancedContentLoader

logger = logging.getLogger(__name__)


class OverallState(TypedDict):
    chunks: List[Document]
    intermediate_summaries: Annotated[List[str], operator.add]
    final_summary: str
    method: Literal["stuff", "map_reduce", "refine"]


class SummaryState(TypedDict):
    content: str
    existing_summary: str  # For refine method


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
        self.graph = StateGraph(OverallState)
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
        
        self.log_verbose(f"Initializing SummarizationAssistant with model: {model_name}")
        
        # Load prompt templates and wrap them in ChatPromptTemplate
        self.stuff_prompt = self._create_chat_prompt(
            f"{self.prompt_info}_stuff",
            self.config,
            self.language_choice
        )
        self.map_prompt = self._create_chat_prompt(
            f"{self.prompt_info}_map",
            self.config,
            self.language_choice
        )
        self.map_reduce_prompt = self._create_chat_prompt(
            f"{self.prompt_info}_map_reduce",
            self.config,
            self.language_choice
        )
        self.reduce_prompt = self._create_chat_prompt(
            "reduce_template",
            self.config,
            self.language_choice
        )
        self.refine_prompt = self._create_chat_prompt(
            f"{self.prompt_info}_refine",
            self.config,
            self.language_choice
        )
        self.log_verbose("Prompt templates loaded successfully")
        
        # Initialize graph_runnable with None
        self.graph_runnable = None
        
        # Call setup_graph() during initialization
        self.setup_graph()

    def _create_chat_prompt(self, prompt_name: str, config: dict, language: str) -> ChatPromptTemplate:
        """Create a ChatPromptTemplate with proper variable mapping."""
        prompt_template = get_prompt_template(prompt_name, config, language)
        
        # Create a ChatPromptTemplate with system message
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates summaries."),
            ("human", prompt_template)
        ])
        
        # Map the incoming 'docs' to 'user_message' in the template
        return chat_prompt.partial(user_message=lambda x: x["docs"])

    def log_verbose(self, message: str):
        if self.verbose:
            logger.info(message)

    async def summarize_stuff(self, state: OverallState) -> dict:
        self.log_verbose("Starting 'stuff' summarization method")
        
        # Handle both Document objects and strings
        if isinstance(state["chunks"][0], Document):
            combined_text = "\n\n".join(
                [chunk.page_content for chunk in state["chunks"]]
            )
        else:
            combined_text = "\n\n".join(state["chunks"])
        
        self.log_verbose(f"Combined text length: {len(combined_text)} characters")
        
        # Use the stuff prompt template
        chain = self.stuff_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "docs": combined_text,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        self.log_verbose(f"'Stuff' summarization completed. Length: {len(summary)}")
        return {"final_summary": summary}
    
    async def generate_map_summary(self, state: dict) -> dict:
        """Generate summary for a single chunk of text."""
        content = state.get("content", "")
        self.log_verbose(
            f"Generating map summary for chunk of length: {len(content)} characters"
        )
        
        # Use the map prompt template
        chain = self.map_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "docs": content,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        self.log_verbose(f"Map summary generated. Length: {len(summary)}")
        return {"intermediate_summaries": [summary]}

    async def reduce_summaries(self, state: OverallState) -> dict:
        self.log_verbose(
            f"Reducing {len(state['intermediate_summaries'])} summaries"
        )
        combined_summaries = "\n\n".join(state["intermediate_summaries"])
        
        # Use the map_reduce prompt template
        chain = self.map_reduce_prompt | self.model | StrOutputParser()
        final_summary = await chain.ainvoke({
            "docs": combined_summaries,
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        self.log_verbose(f"Reduction completed. Length: {len(final_summary)}")
        return {"final_summary": final_summary}

    async def refine_summary(self, state: SummaryState) -> dict:
        """Refine existing summary with new content."""
        self.log_verbose(
            f"Refining summary. New content length: {len(state['content'])}"
        )
        
        # Use the refine prompt template
        chain = self.refine_prompt | self.model | StrOutputParser()
        refined_summary = await chain.ainvoke({
            "docs": state["content"],
            "existing_answer": state["existing_summary"],
            "prompt_info": state.get("prompt_info", self.prompt_info)
        })
        
        self.log_verbose(f"Summary refined. Length: {len(refined_summary)}")
        
        # Check if there are remaining chunks to process
        remaining_chunks = state.get("remaining_chunks", [])
        if remaining_chunks:
            # Get the next chunk
            next_chunk = remaining_chunks[0]
            content = (
                next_chunk.page_content 
                if isinstance(next_chunk, Document) 
                else next_chunk
            )
            # Return state for next iteration
            return {
                "next": "refine_summary",
                "content": content,
                "remaining_chunks": remaining_chunks[1:],
                "existing_summary": refined_summary,
                "prompt_info": state.get("prompt_info", self.prompt_info)
            }
        else:
            # No more chunks, return final summary
            return {"final_summary": refined_summary}

    def route_summaries(self, state: OverallState) -> dict:
        """Route to appropriate summary method based on strategy."""
        method = state["method"]
        self.log_verbose(f"Routing summaries using {method} method")
        
        if method == "map_reduce":
            # For map_reduce, prepare the first chunk
            chunk = state["chunks"][0]
            content = chunk.page_content if isinstance(chunk, Document) else chunk
            # Update state with first chunk and remaining chunks
            return {
                "next": "generate_map_summary",
                "content": content,
                "remaining_chunks": state["chunks"][1:],
                "intermediate_summaries": [],
                "prompt_info": state.get("prompt_info", self.prompt_info)
            }
        elif method == "refine":
            # For refine, prepare the first chunk
            chunk = state["chunks"][0]
            content = chunk.page_content if isinstance(chunk, Document) else chunk
            return {
                "next": "refine_summary",
                "content": content,
                "remaining_chunks": state["chunks"][1:],
                "existing_summary": "",
                "prompt_info": state.get("prompt_info", self.prompt_info)
            }
        else:  # stuff method
            return {
                "next": "summarize_stuff",
                "prompt_info": state.get("prompt_info", self.prompt_info)
            }

    def setup_graph(self):
        self.log_verbose("Setting up summarization graph")
        self.graph = StateGraph(OverallState)

        def get_next_step(state: dict) -> str:
            return state["next"]

        # Add nodes
        self.graph.add_node("route_summaries", self.route_summaries)
        self.graph.add_node("summarize_stuff", self.summarize_stuff)
        self.graph.add_node("generate_map_summary", self.generate_map_summary)
        self.graph.add_node("reduce_summaries", self.reduce_summaries)
        self.graph.add_node("refine_summary", self.refine_summary)

        # Add edges
        self.graph.add_edge(START, "route_summaries")
        
        # Add conditional edges from router
        self.graph.add_conditional_edges(
            "route_summaries",
            get_next_step,
            {
                "summarize_stuff": "summarize_stuff",
                "generate_map_summary": "generate_map_summary",
                "refine_summary": "refine_summary"
            }
        )

        # Add remaining edges
        self.graph.add_edge("generate_map_summary", "reduce_summaries")
        self.graph.add_edge("reduce_summaries", END)
        self.graph.add_edge("summarize_stuff", END)
        self.graph.add_edge("refine_summary", "refine_summary")  # Self-loop for iteration
        self.graph.add_edge("refine_summary", END)

        # Compile the graph
        self.graph_runnable = self.graph.compile()
        self.log_verbose("Summarization graph setup completed")
            
    async def summarize(
        self,
        content: Union[str, List[str], List[Document]],
        method: str = None,
        prompt_info: str = None,
        language: str = "english"
    ) -> dict:
        self.log_verbose(f"Starting summarization process using {method} method")
        self.log_verbose("Loading and splitting documents")

        if isinstance(content, str):
            # Assume it's a file path or URL
            chunks = self.content_loader.load_and_split_document(content)
        elif isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                # List of file paths or URLs
                chunks = self.content_loader.load_and_split_document(content)
            elif all(isinstance(item, Document) for item in content):
                # List of Documents
                chunks = self.content_loader.split_documents(
                    content,
                    self.chunk_size,
                    self.chunk_overlap
                )
            else:
                raise ValueError(
                    "Content list must contain either all strings or all Documents"
                )
        else:
            raise ValueError(
                "Content must be either a string, list of strings, or list of Documents"
            )

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
        self.log_verbose(
            f"Summarization completed. Final summary length: "
            f"{len(result['final_summary'])} characters"
        )
        return result
