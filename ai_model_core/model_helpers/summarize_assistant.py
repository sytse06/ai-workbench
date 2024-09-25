# summarization_assistant_v2.py
import logging
from typing import TypedDict, List, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ai_model_core import get_model, get_prompt_template
from ai_model_core.config.settings import load_config
from ai_model_core.utils import EnhancedContentLoader

logger = logging.getLogger(__name__)


class State(TypedDict):
    input: str
    chunks: List[str]
    intermediate_summaries: List[str]
    final_summary: str
    all_actions: Annotated[List[str], add]


class SummarizationAssistant:
    def __init__(
        self,
        model_name="Ollama (LLama3.1)",
        chunk_size=500,
        chunk_overlap=200,
        max_tokens=1000,
        temperature=0.4,
        chain_type="stuff",
        language="english",
        verbose=False
    ):
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.language = language
        self.config = load_config()
        self.chain_type = chain_type
        self.verbose = verbose
        self.content_loader = EnhancedContentLoader(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.graph = StateGraph(State)

    def log_verbose(self, message: str):
        if self.verbose:
            logger.info(message)

    def load_and_split_document(self, file_path: str) -> List[str]:
        documents = self.content_loader.load_and_split_document(file_path)
        self.log_verbose(f"Loaded and split document into {len(documents)} chunks")
        return [doc.page_content for doc in documents]

    def summarize_stuff(self, chunks: List[str]) -> str:
        combined_text = "\n\n".join(chunks)
        summarize_prompt = get_prompt_template("summarize_stuff", self.config)

        chain = (
            summarize_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        self.log_verbose("Summarizing using 'stuff' method")
        self.log_verbose(f"Prompt: {summarize_prompt}")

        result = chain.invoke({"text": combined_text})

        self.log_verbose(f"Summary result: {result}")

        return result

    def summarize_map_reduce(self, chunks: List[str]) -> str:
        map_prompt = get_prompt_template("summarize_map", self.config)
        reduce_prompt = get_prompt_template("summarize_map_reduce", self.config)

        map_chain = (
            map_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens // 2
            )
        )

        self.log_verbose("Summarizing using 'map_reduce' method")
        self.log_verbose(f"Map prompt: {map_prompt}")

        intermediate_summaries = []
        for i, chunk in enumerate(chunks):
            self.log_verbose(f"Processing chunk {i+1}/{len(chunks)}")
            summary = map_chain.invoke({"chunk": chunk})
            intermediate_summaries.append(summary)
            self.log_verbose(f"Intermediate summary {i+1}: {summary}")

        reduce_chain = (
            reduce_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        self.log_verbose(f"Reduce prompt: {reduce_prompt}")

        summaries = "\n\n".join(intermediate_summaries)
        result = reduce_chain.invoke({"summaries": summaries})

        self.log_verbose(f"Final summary: {result}")

        return result

    def summarize_refine(self, chunks: List[str]) -> str:
        refine_prompt = get_prompt_template("summarize_refine", self.config)

        chain = (
            refine_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        self.log_verbose("Summarizing using 'refine' method")
        self.log_verbose(f"Refine prompt: {refine_prompt}")

        current_summary = ""
        for i, chunk in enumerate(chunks):
            self.log_verbose(f"Processing chunk {i+1}/{len(chunks)}")
            if i == 0:
                current_summary = chain.invoke({
                    "text": chunk,
                    "existing_summary": ""
                })
            else:
                current_summary = chain.invoke({
                    "text": chunk,
                    "existing_summary": current_summary
                })
            self.log_verbose(f"Current summary after chunk {i+1}: {current_summary}")

        return current_summary

    def load_document(self, state):
        file_path = state["input"]
        self.log_verbose(f"Loading document: {file_path}")
        chunks = self.load_and_split_document(file_path)
        return {"chunks": chunks}

    def summarize_chunks(self, state):
        chunks = state["chunks"]
        self.log_verbose(f"Summarizing {len(chunks)} chunks using {self.chain_type} method")
        if self.chain_type == "stuff":
            summary = self.summarize_stuff(chunks)
        elif self.chain_type == "map_reduce":
            summary = self.summarize_map_reduce(chunks)
        elif self.chain_type == "refine":
            summary = self.summarize_refine(chunks)
        else:
            raise ValueError(f"Unknown chain type: {self.chain_type}")
        return {"final_summary": summary}

    def setup_graph(self):
        self.graph.add_node("load_document", self.load_document)
        self.graph.add_node("summarize_chunks", self.summarize_chunks)

        self.graph.add_edge("load_document", "summarize_chunks")
        self.graph.add_edge("summarize_chunks", END)

        self.graph_runnable = self.graph.compile()

    async def summarize(self, file_path: str) -> str:
        self.log_verbose(f"Summarizing file: {file_path}")
        self.log_verbose(f"Using chain type: {self.chain_type}")

        if not hasattr(self, 'graph_runnable'):
            self.setup_graph()

        result = await self.graph_runnable.ainvoke({"input": file_path})
        return result["final_summary"]