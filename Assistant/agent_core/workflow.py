import uuid
from typing import Dict
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver

# Import core modules
from .responder import Responder
from .executor import TaskExecutor
from .planner import TaskPlanning, validation_router


# ==========================================================
#  STATE DEFINITION - Shared across all steps
# ==========================================================
class State(TypedDict):
    thread_id: str
    run_id: str
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str


# ==========================================================
#  WORKFLOW DEFINITION
# ==========================================================
class Workflow:
    def __init__(self, base_llm, plan_structure_llm):
        self.base_llm = base_llm
        self.plan_structure_llm = plan_structure_llm
        self.db: Session | None = None
        self.failed = False  # workflow-level failure flag
        self.graph = self._build_workflow()

        # Core step modules
        self.task_planning = TaskPlanning(self.plan_structure_llm)
        self.task_executor = TaskExecutor(self.base_llm, self.plan_structure_llm)
        self.response = Responder(self.base_llm)

    # ------------------------------------------------------
    #  DB setter
    # ------------------------------------------------------
    def set_db(self, db: Session):
        self.db = db

    # ------------------------------------------------------
    #  SAFE EXECUTION WRAPPER
    # ------------------------------------------------------
    def _safe_node_call(self, func, state: State):
        """
        Executes a node safely. If any node fails, marks the workflow as failed,
        returns a default response, and prevents further node execution.
        """
        print(f"🧩 Executing node: {func.__name__}")

        # Stop if workflow has already failed
        if self.failed:
            print("⚠️ Skipping node execution because workflow has failed previously.")
            return state
        try:
            return func(self.db, state)
        except Exception as e:
            print(f"[Workflow] ❌ Error in {func.__name__}: {e}")
            self.failed = True  # mark workflow as failed

            # Build fallback response safely
            if not isinstance(state.get("response"), dict):
                state["response"] = "Sorry, something went wrong while processing your request."
            state["outputs"] = {"error": str(e)}
            return state

    # ------------------------------------------------------
    #  WORKFLOW GRAPH
    # ------------------------------------------------------
    def _build_workflow(self):
        """
        Create the LangGraph state machine for multi-step execution.
        """
        graph_builder = StateGraph(State)

        # Add all workflow nodes
        graph_builder.add_node("task_planning", self._node_task_planning)
        graph_builder.add_node("validate_plan", self._node_validate_plan)
        graph_builder.add_node("run_plan", self._node_run_plan)
        graph_builder.add_node("generate_response", self._node_generate_response)
        graph_builder.add_node("direct_response", self._node_direct_response)

        # Define flow connections
        graph_builder.set_entry_point("task_planning")
        graph_builder.add_edge("task_planning", "validate_plan")
        graph_builder.add_conditional_edges("validate_plan", validation_router)
        graph_builder.add_edge("run_plan", "generate_response")

        # Define terminal nodes
        graph_builder.set_finish_point("generate_response")
        graph_builder.set_finish_point("direct_response")

        # Add in-memory checkpoint for state persistence
        checkpointer = InMemorySaver()
        return graph_builder.compile(checkpointer=checkpointer)

    # ------------------------------------------------------
    #  NODE WRAPPERS
    # ------------------------------------------------------
    def _node_task_planning(self, state: State):
        return self._safe_node_call(self.task_planning.task_planning, state)

    def _node_validate_plan(self, state: State):
        return self._safe_node_call(self.task_planning.validate_plan, state)

    def _node_run_plan(self, state: State):
        return self._safe_node_call(self.task_executor.run_plan, state)

    def _node_generate_response(self, state: State):
        return self._safe_node_call(self.response.generate_response, state)

    def _node_direct_response(self, state: State):
        return self._safe_node_call(self.response.direct_response, state)
