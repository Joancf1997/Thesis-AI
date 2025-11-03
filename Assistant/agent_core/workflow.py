import uuid
from typing import Dict
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
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

class Workflow:
    def __init__(self, base_llm, plan_structure_llm):
        self.base_llm = base_llm
        self.plan_structure_llm = plan_structure_llm
        self.graph = self._build_workflow()
        
        # Functions - Steps 
        self.task_planning = TaskPlanning(self.plan_structure_llm)
        self.task_executor = TaskExecutor(self.base_llm, self.plan_structure_llm)
        self.response = Responder(self.base_llm)

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
        """Wrapper to call planner module."""
        return self.task_planning.task_planning(state)
         
    def _node_validate_plan(self, state: State):
        """Wrapper to call plan validator."""
        return self.task_planning.validate_plan(state)

    def _node_run_plan(self, state: State):
        """Wrapper to execute plan with adaptive analysis."""
        return self.task_executor.run_plan(state)

    def _node_generate_response(self, state: State):
        """Wrapper to generate final answer from executed plan."""
        return self.response.generate_response(state)

    def _node_direct_response(self, state: State):
        """Wrapper to handle direct LLM answer (no tools)."""
        return self.response.direct_response(state)

    # ------------------------------------------------------
    #  ENTRYPOINT
    # ------------------------------------------------------
    def get_request(self, session_id: uuid.UUID):
        """
        Returns a compiled workflow request with a thread-based session.
        """
        session_config = {"configurable": {"thread_id": str(session_id)}}
        return self.graph, session_config
