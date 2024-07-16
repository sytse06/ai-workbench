import os
import requests
import base64
import yaml
from tqdm import tqdm

def load_config():
    # Construct path to config.yaml relative to the current script
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    
    # Load config from config.yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Check if 'directories' section exists in config
    if 'directories' not in config:
        raise KeyError("Config file is missing 'directories' section")
    
    # Adjust directory paths based on the config
    config['directories']['input_directory'] = os.path.join(os.path.dirname(__file__), config['directories']['input_directory'])
    config['directories']['output_directory'] = os.path.join(os.path.dirname(__file__), config['directories']['output_directory'])
    config['directories']['test_directory'] = os.path.join(os.path.dirname(__file__), config['directories']['test-directory'])
    
    # Validate directory paths
    for directory_key in ['input_directory', 'output_directory', 'test_directory']:
        directory_path = config['directories'][directory_key]
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    return config
    
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_with_llava(image_path):
    base64_image = encode_image_to_base64(image_path)
    input_file_name = os.path.basename(image_path)
    
    response = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'llava',
            'prompt': f'''Analyze this image (original filename: {input_file_name}) and provide the following information in Markdown format:
            1. The printed text in the image (if any)
            2. The handwritten notes or annotations in the image (if any)
            
            Format your response as follows:
            # Original File: {input_file_name}
            
            # Printed Text
            [Printed text content here]
            
            # Handwritten Notes
            [Handwritten notes content here]
            
            If either printed text or handwritten notes are not present, include the heading but leave the content blank.''',
            'images': [base64_image]
        })
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}, {response.text}"

def batch_process_directory(config):
    input_directory = config['directories']['input_directory']
    output_directory = config['directories']['output_directory']
    test_directory = config['directories']['test-directory']
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_directory, image_file)
        llava_output = process_image_with_llava(image_path)
        
        output_file = os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(llava_output)

# Process files in input_directory

if __name__ == "__main__":
    config = load_config()
    batch_process_directory(config)