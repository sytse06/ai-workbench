# summarization_assistant_v2.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Dict
from operator import add
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from ai_model_core import get_model, get_embedding_model, get_prompt_template, get_system_prompt, _format_history, load_document, load_web_content, split_text, create_vectorstore
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
class State(TypedDict):
    input: str
    chunks: List[str]
    intermediate_summaries: List[str]
    final_summary: str
    all_actions: Annotated[List[str], add]

class SummarizationAssistant:
    def __init__(self, model_name="Ollama (LLama3.1)", chunk_size=500, chunk_overlap=200, max_tokens=1000, chain_type="stuff"):
        self.model = get_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph = StateGraph(State)
        self.config = load_config()
        self.chain_type = chain_type
        self.setup_graph()

    def load_and_split_document(self, file_path: str) -> List[str]:
        document = load_document(file_path)
        chunks = split_text(document[0].page_content, self.chunk_size, self.chunk_overlap)
        return chunks

    def summarize_stuff(self, chunks: List[str]) -> str:
        combined_text = "\n\n".join(chunks)
        summarize_prompt = get_prompt_template("summarize_stuff", self.config)
        
        chain = (
            summarize_prompt 
            | self.model.bind(temperature=self.temperature, max_tokens=self.max_tokens)
            | StrOutputParser()
        )
        
        return chain.invoke({"text": combined_text})

    def summarize_map_reduce(self, chunks: List[str]) -> str:
        map_prompt = get_prompt_template("summarize_map", self.config)
        reduce_prompt = get_prompt_template("summarize_map_reduce", self.config)
        
        map_chain = (
            map_prompt 
            | self.model.bind(temperature=self.temperature, max_tokens=self.max_tokens // 2)
            | StrOutputParser()
        )
        
        intermediate_summaries = [map_chain.invoke({"chunk": chunk}) for chunk in chunks]
        
        reduce_chain = (
            reduce_prompt 
            | self.model.bind(temperature=self.temperature, max_tokens=self.max_tokens)
            | StrOutputParser()
        )
        
        return reduce_chain.invoke({"summaries": "\n\n".join(intermediate_summaries)})

    def summarize_refine(self, chunks: List[str]) -> str:
        refine_prompt = get_prompt_template("summarize_refine", self.config)
        
        chain = (
            refine_prompt 
            | self.model.bind(temperature=self.temperature, max_tokens=self.max_tokens)
            | StrOutputParser()
        )
        
        current_summary = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                current_summary = chain.invoke({"text": chunk, "existing_summary": ""})
            else:
                current_summary = chain.invoke({"text": chunk, "existing_summary": current_summary})
        
        return current_summary

    async def summarize(self, file_path: str) -> str:
        chunks = self.load_and_split_document(file_path)
        
        if self.chain_type == "stuff":
            return self.summarize_stuff(chunks)
        elif self.chain_type == "map_reduce":
            return self.summarize_map_reduce(chunks)
        elif self.chain_type == "refine":
            return self.summarize_refine(chunks)
        else:
            raise ValueError(f"Unknown chain type: {self.chain_type}")