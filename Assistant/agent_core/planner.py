from pydantic import Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict


class ArgPair(TypedDict):
    """Key-value pair for tool arguments."""
    key: str
    value: str

class TaskResponseFormatter(TypedDict):
    """Single step in the task plan."""
    task: str = Field(description="Name of the tool to run")
    id: str = Field(description="Unique identifier for the task")
    dep: List[str] = Field(description="List of dependent task IDs")
    args: Optional[List[ArgPair]] = Field(description="Arguments for this tool")
    analyze_answer: Optional[bool] = Field(default=False, description="If True, trigger LLM analysis after this step")
    analyze_target_property: Optional[str] = Field(default=None, description="Specific property of the output to analyze")

class Plan(TypedDict):
    """Full structured plan output."""
    plan: List[TaskResponseFormatter] = Field(description="List of tasks to execute in sequence")

# ==========================================================
#  LLM TASK PLANNING
# ==========================================================

def build_planning_prompt(business_context: str, data_sources: str, tools_context: str) -> ChatPromptTemplate:
    """
    Builds the planning prompt that instructs the LLM to create
    a structured multi-step plan using the available tools.
    """
    full_prompt = business_context + data_sources + tools_context
    return ChatPromptTemplate.from_messages([
        ("system", full_prompt),
        ("human", "Question: {question}")
    ])


def task_planning(state, llm, task_planning_prompt, plan_schema):
    """
    Uses the planning LLM to generate a structured plan from a user question.
    """
    try:
        prompt = task_planning_prompt.invoke({"question": state["question"]})
        result = llm.with_structured_output(plan_schema).invoke(prompt)
        print("🧩 Plan generated successfully.")
        return {"plan": result["plan"]}
    except Exception as e:
        print(f"⚠️ Error generating task plan: {e}")
        return {"plan": []}


# ==========================================================
#  PLAN VALIDATION
# ==========================================================

def validate_plan(plan, available_tools):
    """
    Validate that the plan conforms to the required structure and logic.
    Ensures:
    - No duplicate task IDs
    - All dependencies exist
    - All tools are valid
    - Proper field types
    """
    task_ids = set()
    errors = []

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
        if task_name not in available_tools:
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
        return {"validation": False, "errors": errors}

    print("✅ Plan validation passed.")
    return {"validation": True, "errors": []}


# ==========================================================
#  VALIDATION ROUTER
# ==========================================================

def validation_router(state):
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
