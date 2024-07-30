from setuptools import setup, find_packages

setup(
    name='ai_model_interface',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pydantic',
        'openai',
        'anthropic',
    ],
)