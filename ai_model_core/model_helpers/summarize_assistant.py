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
from langgraph.graph import MessagesState, add_messages
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
from ai_model_core.utils import EnhancedContentLoader
logger = logging.getLogger(__name__)

class SummarizationState(TypedDict):
    messages: Annotated[List[str], operator.add]
    chunks: List[Document]
    intermediate_summaries: Annotated[List[str], operator.add]
    final_summary: str
    method: Literal["stuff", "map_reduce", "refine"]
    current_chunk_index: int
    prompt_info: str

class SummaryState(TypedDict):
    content: str
    existing_summary: str  # For refine method

class SummarizationAssistant:
    def __init__(self, model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, temperature: float = 0.4, method: str = "map_reduce", max_tokens: int = 1000, prompt_info: str = "summarize", language_choice: str = "english", verbose: bool = False):
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.graph = StateGraph(SummarizationState)
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

    async def summarize_stuff(self, state: SummarizationState) -> dict:
        logger.info("Starting 'stuff' summarization method")
        
        interaction_info = []
        
        # Handle both Document objects and strings
        if isinstance(state["chunks"][0], Document):
            combined_text = "\n\n".join([chunk.page_content for chunk in state["chunks"]])
        else:
            combined_text = "\n\n".join(state["chunks"])
        
        interaction_info.append(f"Combined text length: {len(combined_text)} characters")
        
        # Get prompt_info from state or use a fallback if it's not in the state
        prompt_info = state.get("prompt_info", self.prompt_info)
        interaction_info.append(f"Using prompt_info: {prompt_info}")
        
        # Create the Langchain chain with the prompt_info and the summarized text
        chain = self.stuff_prompt | self.model | StrOutputParser()
        interaction_info.append("Invoking LLM chain")
        
        # Capture the full prompt
        full_prompt = self.stuff_prompt.format(user_message=combined_text[:500] + "...", prompt_info=prompt_info)
        interaction_info.append(f"Full prompt: {full_prompt[:500]}...")
        
        summary = await chain.ainvoke({
            "user_message": combined_text,   
            "prompt_info": prompt_info
        })
        
        interaction_info.append(f"'Stuff' summarization completed. Summary length: {len(summary)} characters")
        return {
            "final_summary": summary,
            "interaction_info": "\n".join(interaction_info),
            "final_prompt": full_prompt
        }
    
    async def generate_map_summary(self, state: SummarizationState):
        self.log_verbose(f"Generating map summary for chunk {state['current_chunk_index'] + 1}")
    
        chunk = state["chunks"][state["current_chunk_index"]]
        content = chunk.page_content
        prompt_info = state.get("prompt_info", self.prompt_info)
        
        self.log_verbose(f"Chunk length: {len(content)} characters")
        self.log_verbose(f"Using prompt_info: {prompt_info}")

        chain = self.map_prompt | self.model | StrOutputParser()

        try:
            summary = await chain.ainvoke({
                "user_message": content,
                "prompt_info": prompt_info
            })
            self.log_verbose(f"Map summary generated. Length: {len(summary)} characters")
                        
            return {
                "intermediate_summaries": [summary],  
                "current_chunk_index": state["current_chunk_index"] + 1
            }
        except Exception as e:
            self.log_verbose(f"Error in generate_map_summary: {str(e)}")
            raise

    async def reduce_summaries(self, state: SummarizationState):
        self.log_verbose(f"Reducing {len(state['intermediate_summaries'])} intermediate summaries")
        combined_summaries = "\n\n".join(state["intermediate_summaries"])
        prompt_info = state.get("prompt_info", self.prompt_info)
        
        chain = self.reduce_prompt | self.model | StrOutputParser()
        
        try:
            final_summary = await chain.ainvoke({
                "user_message": combined_summaries,
                "prompt_info": prompt_info
            })
            self.log_verbose(f"Reduction completed. Final summary length: {len(final_summary)} characters")
            return {
                "final_summary": final_summary,
                "intermediate_summaries": [],
                "current_chunk_index": len(state["chunks"])
            }
        except Exception as e:
            self.log_verbose(f"Error in reduce_summaries: {str(e)}")
            raise

    async def refine_summary(self, state: SummarizationState):
        self.log_verbose(f"Refining summary. Processing chunk {state['current_chunk_index'] + 1}")
        
        chunk = state["chunks"][state["current_chunk_index"]]
        content = chunk.page_content
        existing_summary = state.get("final_summary", "")
        prompt_info = state.get("prompt_info", self.prompt_info)
        
        self.log_verbose(f"Chunk length: {len(content)} characters")
        self.log_verbose(f"Existing summary length: {len(existing_summary)} characters")
        self.log_verbose(f"Using prompt_info: {prompt_info}")

        chain = self.refine_prompt | self.model | StrOutputParser()

        try:
            refined_summary = await chain.ainvoke({
                "user_message": f"New content: {content}\n\nExisting summary: {existing_summary}",
                "prompt_info": prompt_info
            })
            self.log_verbose(f"Summary refined. New summary length: {len(refined_summary)} characters")

            return {
                "chunks": [],  # Empty list to avoid updating
                "intermediate_summaries": [],  # Empty list to avoid updating
                "final_summary": refined_summary,
                "method": state["method"],
                "current_chunk_index": state["current_chunk_index"] + 1,
                "prompt_info": state["prompt_info"]
            }
        except Exception as e:
            self.log_verbose(f"Error in refine_summary: {str(e)}")
            raise

    def map_summaries(self, state: SummarizationState) -> List[Send]:
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
    
    def router(self, state: SummarizationState) -> dict:
        method = state["method"]
        if method == "stuff":
            return {"next_step": "summarize_stuff"}
        elif method == "map_reduce":
            if state["current_chunk_index"] < len(state["chunks"]):
                return {"next_step": "generate_map_summary"}
            elif len(state["intermediate_summaries"]) > 0:
                return {"next_step": "reduce_summaries"}
            else:
                return {"next_step": END}
        elif method == "refine":
            if state["current_chunk_index"] < len(state["chunks"]):
                return {"next_step": "refine_summary"}
            else:
                return {"next_step": END}
        else:
            raise ValueError(f"Unknown method: {method}")

    def setup_graph(self):
        self.log_verbose("Setting up summarization graph")
        self.graph = StateGraph(SummarizationState)

        # Defined nodes
        self.graph.add_node("router", self.router)
        self.graph.add_node("summarize_stuff", self.summarize_stuff)
        self.graph.add_node("generate_map_summary", self.generate_map_summary)
        self.graph.add_node("reduce_summaries", self.reduce_summaries)
        self.graph.add_node("refine_summary", self.refine_summary)

        # Defined edges
        self.graph.add_edge(START, "router")
        self.graph.add_conditional_edges(
            "router",
            lambda x: x["next_step"],
            {
                "summarize_stuff": "summarize_stuff",
                "generate_map_summary": "generate_map_summary",
                "reduce_summaries": "reduce_summaries",
                "refine_summary": "refine_summary",
                END: END
            }
        )
        self.graph.add_edge("generate_map_summary", "router")
        self.graph.add_edge("reduce_summaries", END)
        self.graph.add_edge("summarize_stuff", END)
        self.graph.add_edge("refine_summary", "router")

        # Compile the graph
        self.graph_runnable = self.graph.compile()
        self.log_verbose("Summarization graph setup completed")
            
    async def summarize(self, content: Union[str, List[str], List[Document]], method: str = None, prompt_info: str = None, language: str = "english") -> dict:
        interaction_info = []
        method = method or self.method
        interaction_info.append(f"Starting summarization process using {method or self.method} method")
        interaction_info.append(f"Loading and splitting documents")

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

        interaction_info.append(f"Processing {len(chunks)} chunks")

        if not self.graph_runnable:
            self.setup_graph()
            
        initial_state = SummarizationState(
            messages=[], 
            chunks=chunks,
            method=method,
            language=language,
            intermediate_summaries=[],
            final_summary="",
            prompt_info=prompt_info or self.prompt_info,
            current_chunk_index=0
        )

        interaction_info.append(f"Using method: {method}")
        interaction_info.append(f"Language: {language}")
        interaction_info.append(f"Prompt info: {initial_state['prompt_info']}")

        try:
            final_state = await self.graph_runnable.ainvoke(initial_state)
            
            result = {
                "final_summary": final_state.get("final_summary", "No summary generated"),
                "method": method, 
                "current_chunk_index": final_state.get("current_chunk_index", len(chunks)),
                "prompt_info": prompt_info or self.prompt_info
            }

            interaction_info.append(f"Summarization completed. Final summary length: {len(result['final_summary'])} characters")

            result['interaction_info'] = "\n".join(interaction_info)

            if self.verbose and 'intermediate_steps' in final_state:
                result['interaction_info'] += "\n\nIntermediate Steps:\n"
                for i, step in enumerate(final_state['intermediate_steps'], 1):
                    result['interaction_info'] += f"\nStep {i}:\n"
                    result['interaction_info'] += f"Input: {step['input'][:100]}...\n"
                    result['interaction_info'] += f"Output: {step['output'][:100]}...\n"

            self.log_verbose(result['interaction_info'])

            return result
        except Exception as e:
            self.log_verbose(f"An error occurred during summarization: {str(e)}")
            raise