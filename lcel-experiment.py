import yaml
import os
import logging
from PIL import Image
import gradio as gr
from typing import List, Union, Any
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import Field, BaseModel, ConfigDict

# Set up logging
logger = logging.getLogger(__name__)

class OllamaModel(BaseModel):
    base_url: str = Field(default="http://localhost:11434")
    ollama_model_name: str = Field(...)
    chat_model: ChatOllama = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the ChatOllama model using the langchain_community package
        self.chat_model = ChatOllama(
            model=self.ollama_model_name,  # model_name is passed to ChatOllama
            base_url=self.base_url   # base_url for the API
        )

    def __call__(self, inputs: dict):
        messages = [
            SystemMessage(content=inputs["system_prompt"]),
            HumanMessage(content=inputs["human_message"])
        ]
        # Add chat history if available
        if "history" in inputs and inputs["history"]:
            for human, ai in inputs["history"]:
                messages.append(HumanMessage(content=human))
                messages.append(AIMessage(content=ai))
        # Add prompt_info if available
        if "prompt_info" in inputs and inputs["prompt_info"]:
            messages.append(HumanMessage(content=inputs["prompt_info"]))
        
        # Use the model to generate a response
        response = self.chat_model(messages)
        return response.content  # Assuming response has a 'content' field

    def as_runnable(self):
        return RunnableParallel(
            {
                "system_prompt": RunnablePassthrough(),
                "human_message": RunnablePassthrough(),
                "history": RunnablePassthrough(),
                "prompt_info": RunnablePassthrough()
            }
        ) | self

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
    logger.info(f"Formatting prompt with system_prompt: {system_prompt}, user_message: {user_message}, prompt_info: {prompt_info}")
    model = get_model(model_choice)
    system_prompt = get_prompt(prompt_info)
    logger.info(f"Model instantiated: {model}, system_prompt: {system_prompt}")
    
    if stream:
        result = []
        async for chunk in model.prompt(formatted_prompt, system_prompt, stream=True):
            result.append(chunk)
        return result
    else:
        result = await model.prompt(formatted_prompt, system_prompt, stream=False)
        if isinstance(result, str):
            return [result]
        else:
            return result

async def prompt_wrapper(message: str, history: List[tuple[str, str]], model_choice: str, prompt_info: str, language_choice: str, history_flag: bool):
    config = load_config()
    system_prompt = get_system_prompt(language_choice, config)
    prompt_template = get_prompt_template(prompt_info, config)

    # Get the appropriate model using the get_model function
    model = get_model(model_choice)

    # Create the ModelRunnable instance
    model_runnable = model.as_runnable()

    # Create the retrieval chain
    retrieval = RunnableParallel(
        {
            "user_message": RunnablePassthrough(),
            "system_prompt": lambda _: system_prompt,
            "prompt_info": lambda _: prompt_info,
            "history": lambda _: history if history_flag else []
        }
    )

    # Create the full chain
    chain = retrieval | prompt_template | model_runnable | StrOutputParser()

    # Run the chain
    result = await chain.ainvoke(message)
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

