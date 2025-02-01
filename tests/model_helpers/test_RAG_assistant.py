import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from pathlib import Path
import asyncio
from typing import List
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# Import the RAGAssistant and shared test utilities
from your_module_path import RAGAssistant
from .shared_test_utils import BaseMessageFormatTestMixin

# Previous mock classes remain the same
class MockEmbeddingModel:
    """Mock embedding model for testing"""
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.dimension for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        return [0.1] * self.dimension

class MockLLM:
    """Mock LLM for testing"""
    def __init__(self, response="Test response"):
        self.response = response
    
    def bind(self, **kwargs):
        return self
    
    async def ainvoke(self, prompt, **kwargs):
        return self.response

@pytest.fixture
def mock_docs():
    return [
        Document(page_content="Test content 1", metadata={"source": "doc1.txt"}),
        Document(page_content="Test content 2", metadata={"source": "doc2.txt"})
    ]
    
@pytest.fixture
def rag_assistant():
    with patch('your_module_path.get_model', return_value=MockLLM()), \
         patch('your_module_path.get_embedding_model', return_value=MockEmbeddingModel()):
        assistant = RAGAssistant(
            model_name="test-model",
            embedding_model="test-embeddings",
            chunk_size=500,
            chunk_overlap=50
        )
        return assistant

class TestRAGAssistantInitialization:
    def test_init_default_values(self):
        """Test initialization with default values"""
        assistant = RAGAssistant()
        assert assistant.chunk_size == 500
        assert assistant.chunk_overlap == 50
        assert assistant.temperature == 0.4
        assert assistant.num_similar_docs == 3
        assert assistant.language == "english"
        
    def test_init_custom_values(self):
        """Test initialization with custom values"""
        assistant = RAGAssistant(
            chunk_size=1000,
            chunk_overlap=100,
            temperature=0.7,
            num_similar_docs=5,
            language="spanish"
        )
        assert assistant.chunk_size == 1000
        assert assistant.chunk_overlap == 100
        assert assistant.temperature == 0.7
        assert assistant.num_similar_docs == 5
        assert assistant.language == "dutch"

class TestRAGMessageFormat(BaseMessageFormatTestMixin):
    """Test message format validation for RAGAssistant using shared test utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_format_functions(self, rag_assistant):
        self.assistant = rag_assistant
        # Map the required functions from RAGAssistant to the expected interface
        self.format_user_message_func = self.assistant.format_user_message
        self.format_assistant_message_func = self.assistant.format_assistant_message
        self._format_history_func = self.assistant._format_history
        self.convert_to_messages_func = self.assistant.convert_to_messages

class TestContentProcessing:
    @pytest.mark.asyncio
    async def test_process_content(self, rag_assistant, mock_docs):
        """Test content processing with mock documents"""
        with patch.object(rag_assistant.content_loader, 'load_and_split_document', 
                         return_value=mock_docs):
            result = rag_assistant.process_content(
                url_input="",
                file_input=["test.txt"]
            )
            assert "Content loaded successfully" in result
            assert rag_assistant.vectorstore is not None
            
    def test_setup_vectorstore(self, rag_assistant, mock_docs):
        """Test vectorstore setup with different retrieval methods"""
        retrieval_methods = ["similarity", "mmr", "similarity_threshold"]
        
        for method in retrieval_methods:
            rag_assistant.retrieval_method = method
            rag_assistant.setup_vectorstore(mock_docs)
            assert isinstance(rag_assistant.vectorstore, FAISS)
            assert rag_assistant.retriever is not None
            
    def test_setup_vectorstore_empty_docs(self, rag_assistant):
        """Test vectorstore setup with empty document list"""
        with pytest.raises(ValueError, match="No documents were loaded"):
            rag_assistant.setup_vectorstore([])

class TestRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_context(self, rag_assistant, mock_docs):
        """Test context retrieval functionality"""
        rag_assistant.setup_vectorstore(mock_docs)
        retrieved_docs = await rag_assistant.retrieve_context("test query")
        assert len(retrieved_docs) > 0
        assert all(isinstance(doc, Document) for doc in retrieved_docs)
        
    @pytest.mark.asyncio
    async def test_retrieve_context_no_vectorstore(self, rag_assistant):
        """Test context retrieval with no vectorstore setup"""
        with pytest.raises(AttributeError):
            await rag_assistant.retrieve_context("test query")

class TestAnswerGeneration:
    def test_generate_answer(self, rag_assistant):
        """Test answer generation with mock context"""
        state = {
            'context': ['Test context'],
            'question': 'Test question',
            'all_actions': []
        }
        result = rag_assistant.generate_answer(state)
        assert 'answer' in result
        assert 'all_actions' in result
        assert 'answer_generated' in result['all_actions']
        
    @pytest.mark.asyncio
    async def test_query_with_custom_prompt(self, rag_assistant, mock_docs):
        """Test querying with custom prompt template"""
        rag_assistant.setup_vectorstore(mock_docs)
        custom_prompt = "Answer this question based on context: {context}\nQ: {question}"
        
        response = await rag_assistant.query(
            question="Test question",
            prompt_template=custom_prompt
        )
        assert response is not None
        assert isinstance(response, str)
        
    @pytest.mark.asyncio
    async def test_query_with_history(self, rag_assistant, mock_docs):
        """Test querying with conversation history"""
        rag_assistant.setup_vectorstore(mock_docs)
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        response = await rag_assistant.query(
            question="Follow-up question",
            history=history
        )
        assert response is not None
        assert isinstance(response, str)

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_query_no_vectorstore(self, rag_assistant):
        """Test querying without setting up vectorstore"""
        with pytest.raises(ValueError, match="Vector store or retriever not set up"):
            await rag_assistant.query("Test question")
            
    def test_invalid_retrieval_method(self, rag_assistant, mock_docs):
        """Test setup with invalid retrieval method"""
        rag_assistant.retrieval_method = "invalid_method"
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            rag_assistant.setup_vectorstore(mock_docs)

class TestIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, rag_assistant):
        """Test complete RAG flow from content processing to answer generation"""
        # Mock document loading
        test_docs = [
            Document(page_content="The capital of France is Paris.", 
                    metadata={"source": "geography.txt"}),
            Document(page_content="Paris is known for the Eiffel Tower.", 
                    metadata={"source": "landmarks.txt"})
        ]
        
        with patch.object(rag_assistant.content_loader, 'load_and_split_document',
                         return_value=test_docs):
            # Process content
            rag_assistant.process_content(
                url_input="",
                file_input=["test.txt"]
            )
            
            # Query the system
            response = await rag_assistant.query(
                question="What is the capital of France?"
            )
            
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
            
class TestStreamingBehavior:
    @pytest.mark.asyncio
    async def test_streaming_generation(self, rag_assistant, mock_docs):
        """Test streaming response generation"""
        rag_assistant.setup_vectorstore(mock_docs)
        response = await rag_assistant.query(
            question="Test question",
            stream=True
        )
        assert response is not None
        assert isinstance(response, str)