import os
import yaml
from pydantic import BaseModel

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_prompt(name: str) -> str:
    base_dir = os.path.dirname(os.getcwd())
    print(base_dir)
    config_path = os.path.abspath(os.path.join(base_dir, "Thesis-AI/config/prompts", f"{name}.txt"))
    with open(config_path, "r") as file:
        return file.read()
    
class LLMConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float
    max_tokens: int

class PromptConfig(BaseModel):
    planning: str
    response: str

class Settings(BaseModel):
    llm: LLMConfig
