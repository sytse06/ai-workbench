# model_helpers/summarize_assistant.py
# Standard library imports
import logging
from typing import List, Literal, TypedDict, Annotated, Union
import operator
from operator import add

# Third-party imports
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langgraph.constants import Send
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from ..shared_utils import (
    get_model,
    get_embedding_model,
    get_prompt_template,
    _format_history
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
    def __init__(self, model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, temperature: float = 0.4, method: str = "map_reduce", max_tokens: int = 1000, prompt_info: str = "summarize", language_choice: str = "english", verbose: bool = False):
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.graph = StateGraph(OverallState)
        self.method = method
        self.temperature = temperature
        self.content_loader = EnhancedContentLoader(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.config = load_config()
        self.prompt_info = prompt_info
        self.language_choice = language_choice
        self.verbose = verbose
        
        self.log_verbose(f"Initializing SummarizationAssistant with model: {model_name}")
        
        # Load prompt templates
        self.stuff_prompt = get_prompt_template(f"{self.prompt_info}_stuff", self.config, self.language_choice)
        self.map_prompt = get_prompt_template(f"{self.prompt_info}_map", self.config, self.language_choice)
        self.map_reduce_prompt = get_prompt_template(f"{self.prompt_info}_map_reduce", self.config, self.language_choice)
        self.reduce_prompt = get_prompt_template("reduce_template", self.config, self.language_choice)
        self.refine_prompt = get_prompt_template(f"{self.prompt_info}_refine", self.config, self.language_choice)
        self.log_verbose("Prompt templates loaded successfully")
        
        # Initialize graph_runnable with None
        self.graph_runnable = None
        
        # Call setup_graph() during initialization
        self.setup_graph()

    def log_verbose(self, message: str):
        if self.verbose:
            logger.info(message)

    async def summarize_stuff(self, state: OverallState) -> dict:
        self.log_verbose("Starting 'stuff' summarization method")
        
        # Handle both Document objects and strings
        if isinstance(state["chunks"][0], Document):
            combined_text = "\n\n".join([chunk.page_content for chunk in state["chunks"]])
        else:
            combined_text = "\n\n".join(state["chunks"])
        
        self.log_verbose(f"Combined text length: {len(combined_text)} characters")
        
        # Get prompt_info from state or use a fallback if it's not in the state
        prompt_info = state.get("prompt_info", self.prompt_info)
        self.log_verbose(f"Using prompt_info: {prompt_info}")
        
        # Create the Langchain chain with the prompt_info and the summarized text
        chain = self.stuff_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({
            "user_message": combined_text,   
            "prompt_info": prompt_info
        })
        
        self.log_verbose(f"'Stuff' summarization completed. Summary length: {len(summary)} characters")
        return {"final_summary": summary}
    
    async def generate_map_summary(self, state: SummaryState) -> dict:
        self.log_verbose(f"Generating map summary for chunk of length: {len(state['content'])} characters")
        chain = self.map_prompt | self.model | StrOutputParser()
        summary = await chain.ainvoke({"text": state["content"]})
        self.log_verbose(f"Map summary generated. Length: {len(summary)} characters")
        return {"intermediate_summaries": [summary]}

    async def reduce_summaries(self, state: OverallState) -> dict:
        self.log_verbose(f"Reducing {len(state['intermediate_summaries'])} intermediate summaries")
        combined_summaries = "\n\n".join(state["intermediate_summaries"])
        chain = self.map_reduce_prompt | self.model | StrOutputParser()
        final_summary = await chain.ainvoke({"text": combined_summaries})
        self.log_verbose(f"Reduction completed. Final summary length: {len(final_summary)} characters")
        return {"final_summary": final_summary}

    async def refine_summary(self, state: SummaryState) -> dict:
        self.log_verbose(f"Refining summary. New content length: {len(state['content'])} characters")
        chain = self.refine_prompt | self.model | StrOutputParser()
        refined_summary = await chain.ainvoke({
            "text": f"New content: {state['content']}\n\nExisting summary: {state['existing_summary']}"
        })
        self.log_verbose(f"Summary refined. New summary length: {len(refined_summary)} characters")
        return {"final_summary": refined_summary}

    def map_summaries(self, state: OverallState) -> List[Send]:
        self.log_verbose(f"Mapping summaries using {state['method']} method")
        if state["method"] == "map_reduce":
            return [
                Send("generate_map_summary", {"content": chunk.page_content})
                for chunk in state["chunks"]
            ]
        elif state["method"] == "refine":
            return [
                Send("refine_summary", {
                    "content": chunk.page_content,
                    "existing_summary": state.get("final_summary", "")
                })
                for chunk in state["chunks"]
            ]
        else:  # stuff method
            return [Send("summarize_stuff", {})]

    def setup_graph(self):
        self.log_verbose("Setting up summarization graph")
        self.graph = StateGraph(OverallState)

        # Define the router function
        def router(state: OverallState) -> dict:
            method = state["method"]
            if method == "stuff":
                return {"next_step": "summarize_stuff", "method": method}
            elif method == "map_reduce":
                return {"next_step": "generate_map_summary", "method": method}
            elif method == "refine":
                return {"next_step": "refine_summary", "method": method}
            else:
                raise ValueError(f"Unknown method: {method}")

        # Define the decision function
        def decide_next_step(state: OverallState) -> str:
            return state["next_step"]

        # Add nodes
        self.graph.add_node("router", router)
        self.graph.add_node("summarize_stuff", self.summarize_stuff)
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
                "generate_map_summary": "generate_map_summary",
                "refine_summary": "refine_summary",
            }
        )
        self.graph.add_edge("generate_map_summary", "reduce_summaries")
        self.graph.add_edge("summarize_stuff", END)
        self.graph.add_edge("reduce_summaries", END)
        self.graph.add_edge("refine_summary", END)

        # Compile the graph
        self.graph_runnable = self.graph.compile()
        self.log_verbose("Summarization graph setup completed")
            
    async def summarize(self, content: Union[str, List[str], List[Document]], method: str = None, prompt_info: str = None, language: str = "english") -> dict:
        self.log_verbose(f"Starting summarization process using {method} method")
        self.log_verbose(f"Loading and splitting documents")

        if isinstance(content, str):
            # Assume it's a file path or URL
            chunks = self.content_loader.load_and_split_document(content)
        elif isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                # List of file paths or URLs
                chunks = self.content_loader.load_and_split_document(content)
            elif all(isinstance(item, Document) for item in content):
                # List of Documents
                chunks = self.content_loader.split_documents(content, self.chunk_size, self.chunk_overlap)
            else:
                raise ValueError("Content list must contain either all strings (file paths/URLs) or all Documents")
        else:
            raise ValueError("Content must be either a string, a list of strings, or a list of Documents")

        self.log_verbose(f"Processing {len(chunks)} chunks")

        if not self.graph_runnable:
            self.setup_graph()

        result = await self.graph_runnable.ainvoke({
            "chunks": chunks,
            "method": method or self.method,  # Use the provided method or the default
            "language": language,
            "intermediate_summaries": [],
            "final_summary": "",
            "prompt_info": prompt_info or self.prompt_info
        })
        self.log_verbose(f"Summarization completed. Final summary length: {len(result['final_summary'])} characters")
        return result