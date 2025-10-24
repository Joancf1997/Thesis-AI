import uuid
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from utils.utils import Settings, load_prompt

# Import core modules
from agent_core.planner import task_planning, validate_plan, validation_router
from agent_core.executor import run_plan
from agent_core.responder import generate_response, direct_response
from agent_core.analyzer import build_plan_update_prompt, analyze_and_update_plan

from utils.tools import Tools

from typing_extensions import TypedDict
from typing import Dict


# ==========================================================
#  STATE DEFINITION
# ==========================================================
class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

# ==========================================================
#  WORKFLOW BUILDER
# ==========================================================

class Workflow:
    """
    Defines the full LangGraph workflow pipeline for the ARDI Agent:
      - Planning
      - Validation
      - Execution
      - Response generation
    """

    def __init__(self, base_llm, plan_structure_llm, business_context, data_sources, tools_planning):
        self.base_llm = base_llm
        self.plan_structure_llm = plan_structure_llm
        self.business_context = business_context
        self.data_sources = data_sources
        self.tools_planning = tools_planning

        # Initialize tools and analyzer prompt
        self.tools = Tools()
        self.plan_update_prompt = build_plan_update_prompt(business_context)

        # Build the graph
        self.graph = self._build_workflow()

    # ------------------------------------------------------
    #  GRAPH SETUP
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
        """Wrapper to call planner module."""
        from agent_core.planner import build_planning_prompt
        prompt = build_planning_prompt(self.business_context, self.data_sources, self.tools_planning)
        return task_planning(state, self.base_llm, prompt, plan_schema=self.plan_structure_llm.output_schema)

    def _node_validate_plan(self, state: State):
        """Wrapper to call plan validator."""
        return validate_plan(state["plan"], self.tools.TASK_FUNCS)

    def _node_run_plan(self, state: State):
        """Wrapper to execute plan with adaptive analysis."""
        return run_plan(
            state=state,
            tools=self.tools,
            validator=lambda s: validate_plan(s["plan"], self.tools.TASK_FUNCS),
            analyzer_fn=lambda **kwargs: analyze_and_update_plan(
                base_llm=self.base_llm,
                plan_structure_llm=self.plan_structure_llm,
                plan_update_prompt=self.plan_update_prompt,
                **kwargs
            )
        )

    def _node_generate_response(self, state: State):
        """Wrapper to generate final answer from executed plan."""
        return generate_response(self.base_llm, state)

    def _node_direct_response(self, state: State):
        """Wrapper to handle direct LLM answer (no tools)."""
        return direct_response(self.base_llm, state)

    # ------------------------------------------------------
    #  ENTRYPOINT
    # ------------------------------------------------------

    def get_request(self, session_id: uuid.UUID):
        """
        Returns a compiled workflow request with a thread-based session.
        """
        session_config = {"configurable": {"thread_id": str(session_id)}}
        return self.graph, session_config
