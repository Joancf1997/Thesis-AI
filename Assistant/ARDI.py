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
from db.insert_dataset import DatasetEntry
from crud.message import create_human_message, create_assistant_message


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
        self.workflow.failed = False
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
    

    def process_dataset_entries(self, db: Session):
        """
        Fetch all dataset_entries from the database and process them sequentially
        through the given Workflow instance.
        """
        thread_id = "63e30d56-e1b5-46c2-a440-661520ee024f"
        # Fetch all entries
        entries = db.query(DatasetEntry).all()
        print(f"📦 Found {len(entries)} dataset entries to process.")

        for i, entry in enumerate(entries, start=1):
            print(f"\n🚀 Processing entry {i}/{len(entries)} | ID: {entry.id}")
            print(entry.user_query)

            human_msg = create_human_message(db, thread_id=thread_id, content=entry.user_query)
            print(human_msg.content)
            execution = self.ask(db, human_msg)

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
            
            create_assistant_message(db, thread_id=thread_id, content=response_text, resp_msg_id=human_msg.id)
            print(response_text)
