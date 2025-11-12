import os
import uuid
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain.chat_models import init_chat_model

# Internal imports
from utils.utils import Settings
from .agent_core.workflow import Workflow
from .agent_core.planner import Plan  

from crud.run import create_run, end_run

# ==========================================================
#  ARDI AGENT
# ==========================================================
class Agent:
    """
    The ARDI Agent orchestrates the LLM-powered analytical workflow.
    It handles initialization, configuration, and user question resolution.
    """

    def __init__(self, settings: Settings):
        load_dotenv()

        # Session & config
        self.settings = settings
        self.llm_config = settings.llm

        # Initialize everything
        self._init_llms()
        self._init_workflow()
        print("ARDI Agent ready...")

    def _init_llms(self):
        """Initialize both the base LLM and structured-output LLM."""
        self.base_llm = init_chat_model(
            model=self.llm_config.model_name,
            model_provider=self.llm_config.provider,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
        )

        # Structured LLM for JSON plan schema
        self.plan_structure_llm = self.base_llm.with_structured_output(Plan)

    # ------------------------------------------------------
    #  WORKFLOW INITIALIZATION
    # ------------------------------------------------------
    def _init_workflow(self):
        """Build the LangGraph workflow with all nodes connected."""
        self.workflow = Workflow(
            base_llm=self.base_llm,
            plan_structure_llm=self.plan_structure_llm
        )
        self.request = self.workflow.graph

    # ------------------------------------------------------
    #  MAIN ENTRYPOINT - Capture User question
    # ------------------------------------------------------
    def ask(self, db: Session, message_obj):
        """
        Executes the full agent pipeline on a user question.
        """

        thread_id = message_obj.thread_id
        message_content = message_obj.content
        session_config = {"configurable": {"thread_id": str(thread_id)}}
        print(f"User question: {message_content}\n")
        execution_result = {}

        run = create_run(db, message_obj.id)
        self.workflow.set_db(db)
        state = {
            "question": message_content, 
            "thread_id": thread_id,
            "run_id": run.id
        }
        for step in self.request.stream(
            state,
            session_config,
            stream_mode="updates"
        ):
            print(f"📍 Step update: {step}")
            key = list(step.keys())[0]         
            execution_result[key] = step[key]
        
        end_run(db, run.id)
        print("\n✅ Agent pipeline finished successfully!\n")
        return execution_result 
