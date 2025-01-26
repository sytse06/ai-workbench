from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage
import gradio as gr

model = ChatOllama(model="llama3.2",
            base_url="http://localhost:11434")

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant": 
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    response = model.invoke(message)
    return response.content

demo = gr.ChatInterface(
    predict,
    type="messages"
)

if __name__ == "__main__":
    demo.launch()