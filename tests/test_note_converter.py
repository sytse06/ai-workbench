import pytest
import os
import tempfile
import yaml
from ai_model_core import encode_image_to_base64, process_image_with_llava, batch_process_directory, load_config

def test_load_config():
    # Create a temporary config file
    config_data = {
        'directories': {
            'input': 'test_input',
            'output': 'test_output',
            'test': 'test_tests'
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(config_data, temp_file)
    
    # Temporarily replace the original config.yaml path
    original_config_path = 'config.yaml'
    os.rename(original_config_path, original_config_path + '.bak')
    os.rename(temp_file.name, original_config_path)
    
    try:
        # Test loading the config
        config = load_config()
        assert config == config_data
    finally:
        # Restore the original config file
        os.remove(original_config_path)
        os.rename(original_config_path + '.bak', original_config_path)

def test_encode_image_to_base64():
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
        temp_file.write(b'dummy image content')
        temp_file.seek(0)
        encoded = encode_image_to_base64(temp_file.name)
        assert isinstance(encoded, str)
        assert len(encoded) > 0

@pytest.mark.parametrize("status_code,expected", [
    (200, "Sample response"),
    (400, "Error: 400,"),
])
def test_process_image_with_llava(mocker, status_code, expected):
    mock_response = mocker.Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {'response': "Sample response"}
    mock_response.text = "Error message"

    mocker.patch('requests.post', return_value=mock_response)
    mocker.patch('note_converter.encode_image_to_base64', return_value='dummy_base64')

    result = process_image_with_llava('dummy_path.jpg')
    assert expected in result

def test_batch_process_directory(tmpdir, mocker):
    # Create a mock config
    mock_config = {
        'directories': {
            'input': str(tmpdir.mkdir("input")),
            'output': str(tmpdir.mkdir("output"))
        }
    }

    # Create dummy image files
    for i in range(3):
        with open(os.path.join(mock_config['directories']['input'], f"image{i}.jpg"), 'w') as f:
            f.write("dummy image content")

    mocker.patch('note_converter.process_image_with_llava', return_value="Dummy LLaVA output")

    batch_process_directory(mock_config)

    output_dir = mock_config['directories']['output']
    assert len(os.listdir(output_dir)) == 3
    for i in range(3):
        assert os.path.exists(os.path.join(output_dir, f"image{i}.md"))