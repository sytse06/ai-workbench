# ai_model_interface/utils.py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from typing import List, Union, Any
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import yaml

logger = logging.getLogger(__name__)

def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config["system_prompt_settings"][language_choice]["system_prompt"]
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt_template(prompt_info: str, config: dict) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate using the prompt_info and config.

    :param prompt_info: The prompt info selected by the user
    :param config: The loaded configuration
    :return: ChatPromptTemplate
    """
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    prompt_text = config['prompts'].get(prompt_info, "")
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{prompt_info}\n\n{user_message}")
    ])

def _format_history(history: List[tuple[str, str]]) -> List[Union[HumanMessage, AIMessage]]:
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    return formatted_history

# Function to load config from a YAML file
def load_config(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_document(file_path: str) -> List[str]:
    """Load document based on file extension."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def load_web_content(url: str) -> List[str]:
    """Load content from a web URL."""
    loader = WebBaseLoader(url)
    return loader.load()

def create_vectorstore(documents: List[str], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """Create a FAISS vectorstore from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(documents, embeddings)

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)