import os
import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Internal imports
from utils.utils import Settings
from .agent_core.workflow import Workflow
from .agent_core.planner import Plan  


# ==========================================================
#  ARDI AGENT
# ==========================================================
class Agent:
    """
    The ARDI Agent orchestrates the LLM-powered analytical workflow.
    It handles initialization, configuration, and user question resolution.
    """

    def __init__(self, settings: Settings, user_id: uuid.UUID, session_id: uuid.UUID):
        load_dotenv()

        # Session & config
        self.settings = settings
        self.user_id = user_id
        self.session_id = session_id
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

        # Compiled workflow + session configuration
        self.request, self.session_config = self.workflow.get_request(self.session_id)

    # ------------------------------------------------------
    #  MAIN ENTRYPOINT - Capture User question
    # ------------------------------------------------------
    def ask(self, question: str):
        """
        Executes the full agent pipeline on a user question.
        """
        print(f"User question: {question}\n")
        execution_trace = []

        for step in self.request.stream(
            {"question": question},
            self.session_config,
            stream_mode="updates"
        ):
            print(f"📍 Step update: {step}")
            execution_trace.append(step)

        print("\n✅ Agent pipeline finished successfully!\n")
        return execution_trace
