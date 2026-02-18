# AI Workbench

A multi-modal AI workbench with six assistants in a single Gradio
interface. My first serious project with LangChain and Gradio — built
to experiment with different LLM capabilities and run them against
real tasks.

Supports local models (Ollama) and cloud providers (OpenAI, Anthropic,
Mistral) from the same interface.

## Assistants

| Assistant | What it does |
|---|---|
| **Chat** | Conversational interface with conversation history |
| **Prompt** | Reusable prompt templates for repeatable tasks |
| **RAG** | Q&A over your own documents (PDF, DOCX, images) |
| **Summarise** | Document summarisation with adjustable depth |
| **Transcription** | Audio and video transcription via Whisper |
| **Vision** | Image analysis and visual Q&A |

## Model support

Local: Llama 3.1, LLaVA, Deepseek R1 (via Ollama)
Cloud: OpenAI, Anthropic, Mistral

## Stack

LangChain · LangGraph · Gradio · FAISS · Whisper · sentence-transformers
· PyMuPDF · yt-dlp

## Quick start

```bash
git clone https://github.com/sytse06/ai-workbench
cd ai-workbench
poetry install
ollama pull llama3.1   # optional, for local models
python main.py
```

Add API keys to `.env` for cloud model access.
