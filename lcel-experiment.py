import yaml
import os
import logging
from PIL import Image
import gradio as gr
from typing import List, Optional, Any, Dict
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.llms.ollama import _OllamaCommon
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import Field, BaseModel, ConfigDict

# Set up logging
logger = logging.getLogger(__name__)

class OllamaModelConfig(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    ollama_model_name: str = Field(...)
    chat_model: ChatOllama = Field(default=None)

    model_config = {
        "arbitrary_types_allowed": True
    }
    
class OllamaModel(BaseChatModel):
    config: OllamaModelConfig
    chat_model: ChatOllama
    
    def __init__(self, **kwargs):
        config = OllamaModelConfig(**kwargs)
        chat_model = ChatOllama(
            model=config.ollama_model_name,
            base_url=config.base_url,
            verbose=True
        )
        super().__init__(config=config, chat_model=chat_model)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Incorporate the logic from your __invoke__ method
        print("Debug - Messages received in _generate:", messages)
        
        history_messages = []
        if "history" in kwargs and kwargs["history"]:
            for entry in kwargs["history"]:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    human, ai = entry
                    history_messages.append(HumanMessage(content=human))
                    history_messages.append(AIMessage(content=ai))
                else:
                    print("Unexpected history entry format:", entry)

        all_messages = messages + history_messages
        response = self.chat_model.generate([all_messages], stop=stop, run_manager=run_manager)
        
        # Convert the response to ChatResult format
        return ChatResult(generations=[
            ChatGeneration(message=gen.message, generation_info=gen.generation_info)
            for gen in response.generations[0]
        ])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        # Similar logic to _generate, but asynchronous
        print("Debug - Messages received in _agenerate:", messages)
        
        history_messages = []
        if "history" in kwargs and kwargs["history"]:
            for entry in kwargs["history"]:
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    human, ai = entry
                    history_messages.append(HumanMessage(content=human))
                    history_messages.append(AIMessage(content=ai))
                else:
                    print("Unexpected history entry format:", entry)

        all_messages = messages + history_messages
        response = await self.chat_model.agenerate([all_messages], stop=stop, run_manager=run_manager)
        
        return ChatResult(generations=[
            ChatGeneration(message=gen.message, generation_info=gen.generation_info)
            for gen in response.generations[0]
        ])

    @property
    def _llm_type(self) -> str:
        return "ollama-chat"
    
def get_model(choice: str, **kwargs):
    if choice == "Ollama (LLama3.1)":
        return OllamaModel(ollama_model_name="llama3.1", **kwargs)
    # Add other model options here
    else:
        raise ValueError(f"Unsupported model choice: {choice}")

# Load prompt templates
# With these functions
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), 'ai_model_interface/config/config.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if 'system' not in config:
            raise KeyError("Config file is missing 'system' section")
        
        if 'directories' not in config['system']:
            raise KeyError("Config file is missing 'directories' section")
        
        if 'prompts' not in config:
            raise KeyError("Config file is missing 'prompts' section")
        
        logger.info("Config loaded successfully from {config_path}")
        print(f"Config loaded successfully from {config_path}")  # Print to terminal
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ValueError(f"Error parsing YAML file: {e}")

config = load_config()

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

def get_system_prompt(language_choice: str, config: dict) -> str:
    try:
        return config.get("system_prompt_settings", {}).get(language_choice, {}).get("system_prompt", "Default system prompt")
    except KeyError:
        logger.error(f"System prompt not found for language: {language_choice}")
        return "Default system prompt"

def get_prompt(language: str, prompt_name: str) -> str:
    return config.get('prompts', {}).get(language, {}).get(prompt_name, "Default prompt")

def get_prompt_list(language: str) -> List[str]:
    return list(config.get('prompts', {}).get(language, {}).keys())

# Function to update prompt list based on language choice
def update_prompt_list(language: str):
    new_prompts = get_prompt_list(language)
    return gr.Dropdown(choices=new_prompts)

# Define comms functions
async def prompt(formatted_prompt: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, stream: bool = False):
    logger.info(f"Formatting prompt with prompt_info: {prompt_info}")
    model = get_model(model_choice)
    system_prompt = get_prompt(prompt_info)
    logger.info(f"Model instantiated: {model}, system_prompt: {system_prompt}")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_prompt)
    ]
    
    if stream:
        result = []
        async for chunk in model.astream(messages, history=history):
            result.append(chunk.content)
        return result
    else:
        result = await model.agenerate([messages], history=history)
        return [result.generations[0][0].message.content]

async def prompt_wrapper(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, language_choice: str, history_flag: bool):
    config = load_config()
    prompt_template = get_prompt_template(prompt_info, config)

    # Get the appropriate model using the get_model function
    model = get_model(model_choice)

    # Create the retrieval chain
    retrieval = RunnableParallel(
        {
            "messages": lambda user_message: prompt_template.format_messages(
                prompt_info=prompt_info,
                user_message=user_message
            ),
            "history": lambda _: history if history_flag else []
        }
    )
    # Create a function to convert the retrieval output to the correct format
    def format_chat_input(retrieval_output):
        messages = retrieval_output["messages"]
        if history_flag:
            for human, ai in retrieval_output["history"]:
                messages.extend([HumanMessage(content=human), AIMessage(content=ai)])
        return messages

    # Create the full chain
    chain = retrieval | model | StrOutputParser()
    
    # Debugging: Inspect what the retrieval chain outputs
    inputs = message
    print("Debug Input before invoking the chain:", inputs)

    # Run the chain
    result = await chain.ainvoke(inputs)
    return result

def clear_chat():
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Langchain Working Bench")

    with gr.Tab("Prompting"):
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    ["Ollama (LLama3.1)"],
                    label="Choose Model",
                    value="Ollama (LLama3.1)"
                )
                language_choice = gr.Dropdown(
                    ["english", "dutch"],
                    label="Choose Language",
                    value="english"
                )
                prompt_info = gr.Dropdown(choices=get_prompt_list(language_choice.value), label="Prompt Template", interactive=True)
                history_flag = gr.Checkbox(label="Include conversation history", value=True)

            with gr.Column(scale=4):
                prompt_chat_bot = gr.Chatbot(height=600, show_copy_button=True)
                prompt_text_box = gr.Textbox(label="Prompt input", placeholder="Type your question here...")
                gr.ChatInterface(
                    fn=prompt_wrapper,
                    chatbot=prompt_chat_bot,
                    textbox=prompt_text_box,
                    additional_inputs=[model_choice, prompt_info, language_choice, history_flag],
                    submit_btn="Submit",
                    retry_btn="🔄 Retry",
                    undo_btn="↩️ Undo",
                    clear_btn="🗑️ Clear",
                )
                    
    language_choice.change(fn=update_prompt_list, inputs=[language_choice], outputs=[prompt_info])

if __name__ == "__main__":
    logger.info("Starting the Gradio interface")
    demo.launch(debug=True)

