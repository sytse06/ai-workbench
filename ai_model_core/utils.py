# ai_model_interface/utils.py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from typing import List, Union, Any
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.schema import Document
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.schema import Document
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import yaml
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
class EnhancedContentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, file_paths: Union[str, List[str]], urls: str = None) -> List[Document]:
        docs = []

        # Process URLs
        if urls:
            docs.extend(self._load_from_urls(urls))

        # Process files
        if file_paths:
            docs.extend(self._load_from_files(file_paths))

        if not docs:
            raise ValueError("No documents were loaded.")

        return docs

    def _load_from_urls(self, urls: str) -> List[Document]:
        docs = []
        urls_list = urls.split("\n")
        for url in urls_list:
            url = url.strip()
            if url:
                print(f"Loading URL: {url}")
                try:
                    loaded_docs = WebBaseLoader(url).load()
                    if loaded_docs:
                        docs.extend(loaded_docs)
                    else:
                        print(f"Warning: URL {url} returned no content.")
                except Exception as e:
                    print(f"Error loading URL {url}: {str(e)}")
        return docs

    def _load_from_files(self, file_paths: Union[str, List[str]]) -> List[Document]:
        docs = []
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.txt':
                docs.extend(TextLoader(file_path).load())
            elif file_extension == '.pdf':
                docs.extend(self.load_pdf(file_path))
            elif file_extension == '.docx':
                docs.extend(Docx2txtLoader(file_path).load())
            else:
                print(f"Unsupported file type: {file_extension}")
        return docs
    
    def load_pdf(self, file_path: str) -> List[Document]:
        docs = []
        pdf = fitz.open(file_path) #fitz = PyMuPDF
        
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            
            if not text.strip():  # If no text was extracted, try OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": file_path
                    }
                ))
            else:
                print(f"Warning: No text extracted from page {page_num + 1}")
        
        return docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def load_and_split_document(self, file_paths: Union[str, List[str]], urls: str = None) -> List[Document]:
        docs = self.load_documents(file_paths, urls)
        return self.split_documents(docs)
    
def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config["system_prompt_settings"][language_choice]["system_prompt"]
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt_template(prompt_info: str, config: dict, language_choice: str = "english") -> ChatPromptTemplate:
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