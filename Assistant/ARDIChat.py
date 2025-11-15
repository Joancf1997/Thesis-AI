import os
import uuid
import logging
from .ARDI import Agent
from pydantic import BaseModel
from sqlalchemy.orm import Session
from utils.utils import load_config, Settings
from crud.message import create_human_message, create_assistant_message

class Question(BaseModel):
    question: str
    thread_id: str

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "../logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "../logs/system.log"))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

class ChatAssistant():
    def __init__(self):
        configure_logging()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../config/settings.yaml")
        settings = Settings(**load_config(config_path))
        self.agent = Agent(settings)

    def ask(self, db: Session,  question: Question):
        human_msg = create_human_message(db, thread_id=question.thread_id, content=question.question)
        print(human_msg.content)
        execution = self.agent.ask(db, human_msg)

        # Extract relevant information to the user 
        # Response 
        resp_obj = execution.get("direct_response") or execution.get("generate_response")
        response_text = None
        if resp_obj:
            response_text = resp_obj["response"] 
            
        # Outputs - Grounded base for the answer 
        run_plan = execution.get("run_plan")
        plan_outputs = []
        if run_plan:
            for output in run_plan["outputs"]:
                plan_outputs.append(run_plan["outputs"][output]) 
        
        create_assistant_message(db, thread_id=question.thread_id, content=response_text, resp_msg_id=human_msg.id)
        response = {
            "response": response_text,
            "outputs": plan_outputs
        }
        return response
    
    def evaluate_dataset(self, db: Session):
        self.agent.process_dataset_entries(db)
        return "Finish dataset evaluation"