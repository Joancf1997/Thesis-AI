from pydantic import Field
from utils.tools import Tools
from sqlalchemy.orm import Session
from utils.utils import load_prompt
from typing import List, Optional, Dict
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from crud.step import create_step, update_step

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

# PLAN STRUCTURE 
class Plan(TypedDict):
    plan: List[TaskResponseFormatter] = Field(description="List of tasks to be execute")

class TaskPlanning:
    def __init__(self, plan_structure_llm):
        self.plan_structure_llm = plan_structure_llm
        self.planning_prompt = self._build_prompt()


    def _build_prompt(self) -> ChatPromptTemplate:
        """
        Builds the planning prompt that instructs the LLM to create
        a structured multi-step plan using the available tools.
        """
        business_context = load_prompt("0.business_context")
        data_sources = load_prompt("1.data_sources_context")
        tools_description = load_prompt("2.tools_planning")
        full_prompt = business_context + data_sources + tools_description
        return ChatPromptTemplate.from_messages([
            ("system", full_prompt),
            ("human", "Question: {question}")
        ])


    def task_planning(self, db: Session, state: State):
        """
        Uses the planning LLM to generate a structured plan from a user question.
        """
        step = create_step(
            db=db,
            run_id=state["run_id"],
            name="Planning",
            input_data=state["question"]
        )
        try:
            prompt = self.planning_prompt.invoke({"question": state["question"]})
            result = self.plan_structure_llm.invoke(prompt)
            update_step(
                db=db,
                step_id=step.id,
                status="Completed",
                output_data=result
            )
            print("🧩 Plan generated successfully.")
            if result == "{}":
                return []
            return {"plan": result["plan"]}
        except Exception as e:
            print(f"⚠️ Error generating task plan: {e}")
            update_step(
                db=db,
                step_id=step.id,
                status="Error",
            )
            return {"plan": []}

    @staticmethod
    def validate_plan(db: Session, state: State):
        """
        Validate that the plan conforms to the required structure and logic.
        Ensures:
        - No duplicate task IDs
        - All dependencies exist
        - All tools are valid
        - Proper field types
        """
        step = create_step(
            db=db,
            run_id=state["run_id"],
            name="Plan Validation",
            input_data=state["plan"]
        )
        tools = Tools()
        task_ids = set()
        errors = []
        plan = state["plan"]

        for task in plan:
            task_id = task.get("id")
            task_name = task.get("task")
            deps = task.get("dep", [])
            args = task.get("args", [])

            # ---- ID check
            if not isinstance(task_id, str):
                errors.append(f"Task ID must be a string: {task_id}")
            elif task_id in task_ids:
                errors.append(f"Duplicate task ID found: {task_id}")
            else:
                task_ids.add(task_id)

            # ---- Tool validity check
            if task_name not in tools.TASK_FUNCS:
                errors.append(f"Invalid task/tool name: {task_name}")

            # ---- Dependency check
            if not isinstance(deps, list):
                errors.append(f"Dependencies must be a list for task {task_id}")
            else:
                for dep in deps:
                    if dep not in task_ids and dep not in [t.get("id") for t in plan]:
                        errors.append(f"Task {task_id} depends on missing task '{dep}'")

            # ---- Args format
            if not isinstance(args, list):
                errors.append(f"Args must be a list for task {task_id}")

        if errors:
            print("❌ Plan validation failed:")
            for e in errors:
                print(f"   - {e}")
            update_step(
                db=db,
                step_id=step.id,
                status="Completed",
                output_data={"validation": False, "errors": errors}
            )
            return {"validation": False, "errors": errors}

        print("✅ Plan validation passed.")
        update_step(
            db=db,
            step_id=step.id,
            status="Completed",
            output_data={"validation": True, "errors": []}
        )
        return {"validation": True, "errors": []}


# ==========================================================
#  VALIDATION ROUTER
# ==========================================================
def validation_router(state: State):
    """
    Determines which node to run next depending on the plan validation result.
    If plan is empty → go direct_response.
    If validation passes → run_plan.
    Otherwise → replan.
    """
    if len(state.get("plan", [])) == 0:
        print("ℹ️ No tools needed → direct response path.")
        return "direct_response"
    elif state.get("validation"):
        print("➡️  Plan validated → executing plan.")
        return "run_plan"
    else:
        print("🔁 Plan invalid → re-generating.")
        return "task_planning"
