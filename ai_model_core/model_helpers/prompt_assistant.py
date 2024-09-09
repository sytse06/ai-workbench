# model_helpers/prompt_assistant.py
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any
from ai_model_core.config.credentials import get_api_key, load_credentials
from ai_model_core.config.settings import load_config, get_prompt_list, update_prompt_list
from ai_model_core import get_model, get_prompt_template, get_system_prompt, _format_history
from langchain.schema.runnable import RunnableParallel
import logging

logger = logging.getLogger(__name__)

class PromptAssistant:
    def __init__(self, model_choice: str, **kwargs):
        self.model = get_model(model_choice, **kwargs)
        self.model_choice = model_choice
        self.config = load_config()
    
    def update_model(self, model_choice: str, **kwargs):
        if self.model_choice != model_choice:
            self.model = get_model(model_choice, **kwargs)
            self.model_choice = model_choice
    
    def _format_history(self, history: List[tuple[str, str]]) -> List[HumanMessage | AIMessage]:
        formatted_history = []
        for user_msg, ai_msg in history:
            formatted_history.append(HumanMessage(content=user_msg))
            formatted_history.append(AIMessage(content=ai_msg))
        return formatted_history

    def _get_prompt_template(self, prompt_info: str, language_choice: str) -> ChatPromptTemplate:
        system_prompt = get_system_prompt(language_choice, self.config)
        prompt_text = self.config['prompts'].get(prompt_info, "")
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{prompt_info}\n\n{user_message}")
        ])

    async def prompt(self, message: str, history: List[tuple[str, str]], prompt_info: str, language_choice: str, history_flag: bool, stream: bool = False):
        prompt_template = get_prompt_template(prompt_info, self.config)
        system_prompt = get_system_prompt(language_choice, self.config)

        # Start with the system message
        messages = [SystemMessage(content=system_prompt)]

        # Add formatted history if history_flag is True
        if history_flag:
            formatted_history = self._format_history(history)
            messages.extend(formatted_history)

        # Format the current user message using the prompt template
        formatted_prompt = prompt_template.format(prompt_info=prompt_info, user_message=message)
        messages.append(HumanMessage(content=formatted_prompt))

        if stream:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        else:
            result = await self.model.agenerate([messages])
            yield result.generations[0][0].text