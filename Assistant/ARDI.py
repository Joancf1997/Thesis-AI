import io
import os
import uuid
from pydantic import Field
from dotenv import load_dotenv
from PIL import Image as PILImage
from typing_extensions import TypedDict
from typing import Dict, List, Optional

from langgraph.graph import StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver


from utils.tools import Tools
from utils.utils import Settings, load_prompt

# === STATE ===
class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

# FUNCTION ARGUMENTS 
class ArgPair(TypedDict):
    key: str
    value: str

# TASK CALL
class TaskResponseFormatter(TypedDict):
    task: str = Field(description="Name of the tool to run")
    id: str = Field(description="Unique identifier for the task")
    dep: List[str] = Field(description="List of task ids this task depends on")
    args: Optional[List[ArgPair]] = Field(description="Arguments as list of (key, value) pairs")

# PLAN STRUCTURE SCHEMA
class Plan(TypedDict):
    plan: List[TaskResponseFormatter] = Field(description="List of tasks to be execute")

class Agent:
    def __init__(self, settings: Settings, user_id: uuid, session_id: uuid):
        self.llm_config = settings.llm
        self.session_id = session_id
        self.session_config = {"configurable": {"thread_id": session_id}}
        self.state: State = {"question": "", "plan": [], "validation": False, "outputs": [], "response": ""}
        self.tools = Tools()
        self.config_prompt()
        self.config_llm()
        self.workflow_config()
        print("welcome, ARDI is all ready to help...")

    def config_prompt(self):
        # Load the prompt templates
        self.business_context = load_prompt('0.business_context')
        self.data_sources = load_prompt('1.data_sources_context')
        self.tools_planning = load_prompt('2.tools_planning')
        self.planning_prompt = self.business_context + self.data_sources + self.tools_planning
        self.response_prompt = self.planning_prompt + load_prompt('4.response_stage')
        self.direct_response_prompt = self.business_context + self.data_sources + load_prompt('3.direct_response')
        self.task_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", self.planning_prompt),
            ("human", "Question: {question}")
        ])
        self.generate_response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.response_prompt)
        ])
        self.generate_direct_response_prompt = ChatPromptTemplate.from_messages([
            ("system", self.direct_response_prompt)
        ])

    def config_llm(self):
        self.base_llm = init_chat_model(
            model=self.llm_config.model_name,
            model_provider=self.llm_config.provider,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
        )
        self.plan_structure_llm = self.base_llm.with_structured_output(Plan)

    def workflow_config(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("task_planning", self.task_planning)
        graph_builder.add_node("validate_plan", self.validate_plan)
        graph_builder.add_node("run_plan", self.run_plan)
        graph_builder.add_node("generate_response", self.generate_response)
        graph_builder.add_node("direct_response", self.direct_response)
        graph_builder.set_entry_point("task_planning")
        graph_builder.add_edge("task_planning", "validate_plan")
        graph_builder.add_conditional_edges("validate_plan", self.validation_router)
        graph_builder.add_edge("run_plan", "generate_response")
        graph_builder.set_finish_point("generate_response")
        graph_builder.set_finish_point("direct_response")

        # Memory (conversation history persistence)
        checkpointer = InMemorySaver()
        self.request = graph_builder.compile(checkpointer=checkpointer)

        # Session config for thread-based memory
        self.session_config = {"configurable": {"thread_id": str(self.session_id)}}

        # image_bytes = self.request.get_graph().draw_mermaid_png()
        # img = PILImage.open(io.BytesIO(image_bytes))
        # img.save("graph_output.pdf", "PDF")

    def ask(self, question: str):
        execution = []
        for step in self.request.stream({"question": question}, self.session_config, stream_mode="updates"):
          execution.append(step)
        return execution

    
    def task_planning(self, state: State):
        prompt = self.task_planning_prompt.invoke({"question": state["question"]})
        result = self.plan_structure_llm.invoke(prompt)
        return {"plan": result['plan']}

    def validation_router(self, state: dict) -> str:
        if len(state.get("plan")) == 0:
            return "direct_response"
        return "run_plan" if state.get("validation") else "task_planning"
    
    def validate_plan(self, state: State):
        task_ids = set()
        errors = []

        for task in state["plan"]:
            task_id = task.get('id')
            task_name = task.get('task')
            deps = task.get('dep', [])
            args = task.get('args', [])
            if not isinstance(task_id, str):
                errors.append(f"Task ID must be a string: {task_id}")
            elif task_id in task_ids:
                errors.append(f"Duplicate task ID found: {task_id}")
            else:
                task_ids.add(task_id)
            if task_name not in self.tools.TASK_FUNCS:
                errors.append(f"Invalid task/tool name: {task_name}")
            if not isinstance(deps, list):
                errors.append(f"Dependencies must be a list for task {task_id}")
        if errors:
            print("Plan validation False..")
            print(errors)
            return {"validation": False}
        return {"validation": True}
    
    def run_plan(self, state: State):
        outputs = {}
        remaining = state["plan"].copy()

        def resolve_args(args_list):
            resolved = {}
            for arg in args_list:
                k = arg['key']
                v = arg['value']
                if isinstance(v, str) and v.startswith("DEP_"):
                    dep_task_id = v[4:]
                    resolved[k] = outputs[dep_task_id]
                else:
                    resolved[k] = v
            return resolved

        while remaining:
            progress = False
            for task in remaining[:]:
                if all(dep in outputs for dep in task['dep']):
                    args = resolve_args(task.get('args', []))
                    func = self.tools.TASK_FUNCS.get(task['task'])
                    result = func(**args)
                    outputs[task['id']] = result
                    remaining.remove(task)
                    progress = True
            if not progress:
                raise RuntimeError("Circular dependency or missing dependencies detected")
        return {"outputs": outputs}

    def direct_response(self, state: State):
      prompt = self.generate_direct_response_prompt.invoke({"question": state["question"]})
      response = self.base_llm.invoke(prompt)
      return {"response": response}

    def generate_response(self, state: State):
      prompt = self.generate_response_prompt.invoke({"question": state["question"], "plan": state["plan"], "tool_outputs": state["outputs"]})
      response = self.base_llm.invoke(prompt)
      return {"response": response}
      
    