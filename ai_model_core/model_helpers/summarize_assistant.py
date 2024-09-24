# summarization_assistant_v2.py
import logging
from typing import TypedDict, List, Annotated
from operator import add
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

    def load_and_split_document(self, file_path: str) -> List[str]:
        documents = self.content_loader.load_and_split_document(file_path)
        if self.verbose:
            logger.info(
                f"Loaded and split document into {len(documents)} chunks"
            )
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

        if self.verbose:
            logger.info("Summarizing using 'stuff' method")
            logger.info(f"Prompt: {summarize_prompt}")

        result = chain.invoke({"text": combined_text})

        if self.verbose:
            logger.info(f"Summary result: {result}")

        return result

    def summarize_map_reduce(self, chunks: List[str]) -> str:
        map_prompt = get_prompt_template("summarize_map", self.config)
        reduce_prompt = get_prompt_template(
            "summarize_map_reduce", self.config
        )

        map_chain = (
            map_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens // 2
            )
        )

        if self.verbose:
            logger.info("Summarizing using 'map_reduce' method")
            logger.info(f"Map prompt: {map_prompt}")

        intermediate_summaries = []
        for i, chunk in enumerate(chunks):
            if self.verbose:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            summary = map_chain.invoke({"chunk": chunk})
            intermediate_summaries.append(summary)
            if self.verbose:
                logger.info(f"Intermediate summary {i+1}: {summary}")

        reduce_chain = (
            reduce_prompt
            | self.model.bind(
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )

        if self.verbose:
            logger.info(f"Reduce prompt: {reduce_prompt}")

        summaries = "\n\n".join(intermediate_summaries)
        result = reduce_chain.invoke({"summaries": summaries})

        if self.verbose:
            logger.info(f"Final summary: {result}")

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

        if self.verbose:
            logger.info("Summarizing using 'refine' method")
            logger.info(f"Refine prompt: {refine_prompt}")

        current_summary = ""
        for i, chunk in enumerate(chunks):
            if self.verbose:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
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
            if self.verbose:
                logger.info(
                    f"Current summary after chunk {i+1}: {current_summary}"
                )

        return current_summary

    async def summarize(self, file_path: str) -> str:
        chunks = self.load_and_split_document(file_path)

        if self.verbose:
            logger.info(f"Summarizing file: {file_path}")
            logger.info(f"Using chain type: {self.chain_type}")

        if self.chain_type == "stuff":
            return self.summarize_stuff(chunks)
        elif self.chain_type == "map_reduce":
            return self.summarize_map_reduce(chunks)
        elif self.chain_type == "refine":
            return self.summarize_refine(chunks)
        else:
            raise ValueError(f"Unknown chain type: {self.chain_type}")