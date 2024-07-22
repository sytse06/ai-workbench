# Link to original repo which enables Ollama models with vision language interaction https://github.com/ycyy/ollama-gradio-webui/tree/main
import gradio as gr
import ollama
import base64
import copy
import os
import yaml

# Initialize prompts and chat memory
PROMPT_LIST = []
VL_CHAT_LIST = []

# List running Ollama models
# model_list = ollama.list()
# model_names = [model['model'] for model in model_list['models']]

# List running Ollama models in interface
def get_running_models():
    try:
        running_models = ollama.ps()
        if 'models' in running_models:
            return [(model['model'], model['model']) for model in running_models['models']]
        else:
            print("Unexpected structure of the response from ollama.ps().")
            return []
    except ollama.exceptions.OllamaError as e:  # Catch specific Ollama-related exceptions
        print(f"Error getting running models: {str(e)}")
        return []
    except Exception as e:  # Catch any other potential errors
        print(f"An unexpected error occurred: {str(e)}")
        return []

# Load settings from config.yaml file
def load_config():
    # Construct path to config.yaml relative to the current script
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    
    try:
        # Load config from config.yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Check if 'system' section exists in config
        if 'system' not in config:
            raise KeyError("Config file is missing 'system' section")
        
        # Check if 'directories' section exists in config
        if 'directories' not in config['system']:
            raise KeyError("Config file is missing 'directories' section")
        
        # Check if 'prompts' section exists in config
        if 'prompts' not in config:
            raise KeyError("Config file is missing 'prompts' section")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

# Example usage
try:
    config = load_config()
    print("Config loaded successfully")
    print(f"Input directory: {config['system']['directories']['input_directory']}")
    print(f"Output directory: {config['system']['directories']['output_directory']}")
    print(f"Vision Assistant prompt loaded: {config['prompts']['Vision Assistant'][:50]}...")
    PROMPT_LIST = list(config['prompts'].keys())
except Exception as e:
    print(f"Error loading config: {e}")
        
# Initialize function
def init():
    VL_CHAT_LIST.clear()
    
def ollama_chat(message, history, model_name, history_flag):
    if not model_name:
        return "Please select a running Ollama model."
    
    messages = []
    if history_flag and history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        partial_message = ""
        for chunk in stream:
            if chunk['message']['content']:
                partial_message += chunk['message']['content']
                yield partial_message
    except Exception as e:
        yield f"Error: {str(e)}"

# Generate agent response
def ollama_prompt(message, history, model_name,prompt_info):
    messages = []
    system_message = {
        'role': 'system', 
        'content': config['prompts'][prompt_info]
    }
    user_message = {
        'role': 'user', 
        'content': message
    }
    messages.append(system_message)
    messages.append(user_message)
    stream = ollama.chat(
        model = model_name,
        messages = messages,       
        stream=True
    )
    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message
        try:
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True
            )
        except Exception as e:
            return f"Error: {str(e)}"
            
# Image upload
def vl_image_upload(image_path,chat_history):
    messsage = {
        "type":"image",
        "content":image_path
    }
    chat_history.append(((image_path,),None))
    VL_CHAT_LIST.append(messsage)
    return None,chat_history

# Submit message
def vl_submit_message(message,chat_history):
    messsage = {
        "type":"user",
        "content":message
    }
    chat_history.append((message,None))
    VL_CHAT_LIST.append(messsage)
    return "",chat_history

# Retry
def vl_retry(chat_history):
    if len(VL_CHAT_LIST)>1:
        if VL_CHAT_LIST[len(VL_CHAT_LIST)-1]['type'] == "assistant":
            VL_CHAT_LIST.pop()
            chat_history.pop()
    return chat_history

# Undo
def vl_undo(chat_history):
    message = ""
    chat_list = copy.deepcopy(VL_CHAT_LIST)
    if len(chat_list)>1:
        if chat_list[len(chat_list)-1]['type'] == "assistant":
            message = chat_list[len(chat_list)-2]['content']
            VL_CHAT_LIST.pop()
            VL_CHAT_LIST.pop()
            chat_history.pop()
            chat_history.pop()
        elif chat_list[len(chat_list)-1]['type'] == "user":
            message = chat_list[len(chat_list)-1]['content']
            VL_CHAT_LIST.pop()
            chat_history.pop()
    return message,chat_history

# Clear chat
def vl_clear():
    VL_CHAT_LIST.clear()
    return None,"",[]

# Submit response
def vl_submit(history_flag,chinese_flag,chat_history):
    if len(VL_CHAT_LIST)>1:
        messages = get_vl_message(history_flag,chinese_flag)
        response = ollama.chat(
            model = "llava:7b-v1.6",
            messages = messages
        )
        result = response["message"]["content"]
        output = {
            "type":"assistant",
            "content":result
        }
        chat_history.append((None,result))
        VL_CHAT_LIST.append(output)
    else:
        gr.Warning('Error retrieving result')
    return chat_history

def get_vl_message(history_flag,chinese_flag):
    messages = []
    if history_flag:
        i=0
        while i<len(VL_CHAT_LIST):
            if VL_CHAT_LIST[i]['type']=="image" and VL_CHAT_LIST[i+1]['type']=="user":
                image_path = VL_CHAT_LIST[i]["content"]
                # Read image file binary data
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                # Convert binary data to base64 string
                base64_string = base64.b64encode(image_data).decode("utf-8")
                content = VL_CHAT_LIST[i+1]["content"]
                chat_message = {
                    'role': 'user', 
                    'content': content,
                    'images':[base64_string]
                }
                messages.append(chat_message)
                i+=2
            elif VL_CHAT_LIST[i]['type']=="assistant":
                assistant_message = {
                    "role":"assistant",
                    "content":VL_CHAT_LIST[i]['content']
                }
                messages.append(assistant_message)
                i+=1
            elif VL_CHAT_LIST[i]['type']=="user":
                user_message = {
                    "role":"user",
                    "content":VL_CHAT_LIST[i]['content']
                }
                messages.append(user_message)
                i+=1
            else:
                i+=1
    else:
        if VL_CHAT_LIST[0]['type']=="image" and VL_CHAT_LIST[-1]['type']=="user":
            image_path = VL_CHAT_LIST[0]["content"]
            # Read image file binary data
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            # Convert binary data to base64 string
            base64_string = base64.b64encode(image_data).decode("utf-8")
            content = VL_CHAT_LIST[-1]["content"]
            chat_message = {
                'role': 'user', 
                'content': content,
                'images':[base64_string]
            }
            messages.append(chat_message)
    if chinese_flag:
        system_message = {
            'role': 'system', 
            'content': 'You are a Helpful Assistant. Please answer the question in English'
        }
        messages.insert(0,system_message)
    return messages

def main():
    choices = get_running_models()
    
    with gr.Blocks(title="Ollama WebUI", fill_height=True) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                model_info = gr.Dropdown(
                    choices = choices, 
                    label="Select a running model"
                    )
                refresh_btn = gr.Button("Refresh Models")
                    
        with gr.Tab():                
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):        
                        history_flag = gr.Checkbox(label="Enable Context")
                    with gr.Column(scale=4):
                        chat_bot = gr.Chatbot(height=600, render=False)
                        text_box = gr.Textbox(scale=4, render=False)
                        chat_interface = gr.ChatInterface(
                            fn=ollama_chat,
                            chatbot=chat_bot,
                            textbox=text_box,
                            additional_inputs=[model_info, history_flag],
                            submit_btn="Submit",
                            retry_btn="🔄 Retry",
                            undo_btn="↩️ Undo",
                            clear_btn="🗑️ Clear",
                            fill_height=True
                        )
                                    
        with gr.Tab("Agent"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_model_info = gr.Dropdown(choices=choices, value="", allow_custom_value=True, label="Model Selection")
                    prompt_info = gr.Dropdown(choices=PROMPT_LIST, value=PROMPT_LIST[0] if PROMPT_LIST else None, label="Agent Selection", interactive=True)
                with gr.Column(scale=4):
                    prompt_chat_bot = gr.Chatbot(height=600, render=False)
                    prompt_text_box = gr.Textbox(scale=4, render=False)
                    gr.ChatInterface(
                        fn=ollama_prompt,
                        chatbot=prompt_chat_bot,
                        textbox=prompt_text_box,
                        additional_inputs=[prompt_model_info, prompt_info],
                        submit_btn="Submit",
                        retry_btn="🔄 Retry",
                        undo_btn="↩️ Undo",
                        clear_btn="🗑️ Clear",
                        fill_height=True
                    )
        with gr.Tab("Vision Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    history_flag = gr.Checkbox(label="Enable Context")
                    chinese_flag = gr.Checkbox(value=True, label="Force Chinese")
                    image = gr.Image(type="filepath")
                with gr.Column(scale=4):
                    chat_bot = gr.Chatbot(height=600)
                    with gr.Row():
                        retry_btn = gr.Button("🔄 Retry")
                        undo_btn = gr.Button("↩️ Undo")
                        clear_btn = gr.Button("🗑️ Clear")
                    with gr.Row():
                        message = gr.Textbox(show_label=False, container=False, scale=5)
                        submit_btn = gr.Button("Submit", variant="primary", scale=1)
            image.upload(fn=vl_image_upload, inputs=[image, chat_bot], outputs=[image, chat_bot])
            submit_btn.click(fn=vl_submit_message, inputs=[message, chat_bot], outputs=[message, chat_bot]).then(fn=vl_submit, inputs=[history_flag, chinese_flag, chat_bot], outputs=[chat_bot])
            retry_btn.click(fn=vl_retry, inputs=[chat_bot], outputs=[chat_bot]).then(fn=vl_submit, inputs=[history_flag, chinese_flag, chat_bot], outputs=[chat_bot])
            undo_btn.click(fn=vl_undo, inputs=[chat_bot], outputs=[message, chat_bot])
            clear_btn.click(fn=vl_clear, inputs=[], outputs=[image, message, chat_bot])
            
        refresh_btn.click(fn=refresh_models, inputs=[], outputs=[model_info, prompt_model_info])
            
        demo.load(fn=init)
        
    return demo

def refresh_models():
    new_choices = get_running_models()
    return gr.Dropdown.update(choices=new_choices), gr.Dropdown.update(choices=new_choices)
    
if __name__ == "__main__":
    demo = main()
    demo.launch(share=False)