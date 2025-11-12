import json
import copy
import pandas as pd
from typing import Dict, Any
from utils.tools import Tools
from .planner import TaskPlanning
from sqlalchemy.orm import Session
from utils.utils import load_prompt
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from crud.step import create_step, update_step
from crud.tool import create_tool_call, update_tool_call


class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

# ====================================
#  UTILITY HELPERS
# ====================================

def make_serializable(value):
    """Recursively convert pandas/numpy objects to JSON-safe formats."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    elif isinstance(value, (list, tuple)):
        return [make_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    else:
        return value


def extract_property(obj, prop_path):
    """Safely extract nested property (supports dot notation and list indices)."""
    try:
        value = obj
        for part in prop_path.split("."):
            if isinstance(value, list) and part.isdigit():
                value = value[int(part)]
            else:
                value = value[part]
        return value
    except (KeyError, IndexError, TypeError):
        raise KeyError(f"Property '{prop_path}' not found in object: {obj}")


def cast_arg(value, expected_type):
    """Safely cast argument according to expected type."""
    if expected_type is None:
        return value
    try:
        if expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == bool:
            return bool(value)
        elif expected_type == list and isinstance(value, str):
            return json.loads(value)
        elif expected_type == str:
            return str(value)
        return value
    except Exception:
        raise ValueError(f"Failed to cast value '{value}' to {expected_type.__name__}")

class TaskExecutor:
    def __init__(self, base_llm, structure_llm):
        self.base_llm = base_llm
        self.plan_structure_llm = structure_llm
        self.tools = Tools()
        self.analyze_prompt_plan = self._build_analyze_prompts() 

    # ====================================
    #  MAIN EXECUTION LOGIC
    # ====================================
    def run_plan(self, db: Session, state: State):
        """
        Executes the analytical plan step by step, resolving dependencies,
        property extraction, type casting, and invoking the LLM-based analyzer
        when 'analyze_answer' is set to True.

        Args:
            state (dict): The shared workflow state containing the plan and question.
        """

        step = create_step(
            db=db,
            run_id=state["run_id"],
            name="Plan Execution",
            input_data=state["plan"]
        )
        outputs = {}
        plan_versions = [copy.deepcopy(state["plan"])]
        remaining = copy.deepcopy(state["plan"])

        TOOL_ARG_TYPES = {
            "get_segment_description": {"segment_id": int},
            "get_segment_engagement_stats": {"segment_id": int},
            "get_topic_transitions": {"segment_id": int, "top_n": int},
            "get_next_topic_prediction": {"segment_id": int, "current_topic": str, "top_n": int},
            "get_segment_regions": {"segment_id": int, "top_n": int},
            "get_segment_time_activity": {"segment_id": int},
            "get_segment_activity_by_day_part": {"segment_id": int},
            "get_segment_articles_by_time": {"segment_id": int, "start_hour": int, "end_hour": int},
            "get_segment_engage_docs": {"segment_id": int},
            "get_segment_not_engage_docs": {"segment_id": int},
            "get_segment_high_rep_docs": {"segment_id": int},
            "get_articles_info": {"articles_ids": list},
            "get_top_recent_articles": {"articles_ids": list, "top": int},
            "get_unique_clusters": {"articles_ids": list},
            "get_news_topics_info": {"topics_id": list},
            "get_news_topics_high_docs": {"topic_id": int},
            "get_news_topics_low_docs": {"topic_id": int},
        }

        def resolve_args(task):
            """Resolve dependency references, extract properties, and cast types."""
            args_list = task.get("args", [])
            resolved = {}

            for arg in args_list:
                k = arg["key"]
                v = arg["value"]
                prop = arg.get("property")

                # Resolve dependency
                if isinstance(v, str) and v.startswith("DEP_"):
                    dep_task_id = v[4:]
                    if dep_task_id not in outputs:
                        update_step(
                            db=db,
                            step_id=step.id,
                            status="Error",
                            output_data=f"Dependency '{dep_task_id}' not yet available for argument '{k}'."
                        )
                        raise ValueError(f"Dependency '{dep_task_id}' not yet available for argument '{k}'.")
                    dep_output = outputs[dep_task_id]
                    if prop:
                        dep_output = extract_property(dep_output, prop)
                    resolved[k] = dep_output
                else:
                    resolved[k] = v

                # Type casting
                expected_type = TOOL_ARG_TYPES.get(task["task"], {}).get(k)
                if expected_type:
                    resolved[k] = cast_arg(resolved[k], expected_type)

            return resolved

        # =========================================
        #  MAIN EXECUTION LOOP
        # =========================================
        while remaining:
            progress = False

            for task in remaining[:]:
                if all(dep in outputs for dep in task.get("dep", [])):
                    args = resolve_args(task)
                    func = self.tools.TASK_FUNCS.get(task["task"])
                    if not func:
                        update_step(
                            db=db,
                            step_id=step.id,
                            status="Error",
                            output_data=f"Unknown tool: {task['task']}"
                        )
                        raise ValueError(f"Unknown tool: {task['task']}")

                    # Execute the tool
                    tool_call = create_tool_call(
                        db=db,
                        step_id=step.id,
                        tool_name=task["task"],
                        input_data=args
                    )
                    result = func(**args)
                    result_serializable = make_serializable(result)
                    update_tool_call(
                        db=db,
                        tool_call_id=tool_call.id,
                        status="success",
                        output_data=result_serializable
                    )
                    outputs[task["id"]] = result_serializable
                    remaining.remove(task)
                    progress = True

                    print(f"✅ Executed: {task['task']} | ID: {task['id']}")

                    # --------------------------------
                    #  Handle analyze_answer flag
                    # --------------------------------
                    if task.get("analyze_answer", False):
                        target_prop = task.get("analyze_target_property")
                        if target_prop:
                            try:
                                target_value = extract_property(result_serializable, target_prop)
                            except Exception as e:
                                print(f"⚠️ Could not extract property '{target_prop}': {e}")
                                target_value = result_serializable
                        else:
                            target_value = result_serializable

                        print(f"🧠 Triggering LLM analysis for '{task['id']}'...")

                        # Call analyzer LLM to update the plan dynamically
                        new_plan = self._analyze_and_update_plan(
                            question=state["question"],
                            plan=remaining,
                            latest_output=target_value,
                            previous_outputs=outputs
                        )

                        validated = TaskPlanning.validate_plan(db, state)
                        if not validated.get("validation", False):
                            print("⚠️ New plan failed validation, skipping update.")
                            continue

                        plan_versions.append(copy.deepcopy(new_plan))
                        print(f"🔄 Plan updated → version {len(plan_versions)}")
                        remaining = copy.deepcopy(new_plan)
                        break  # restart loop with updated plan

            if not progress:
                update_step(
                    db=db,
                    step_id=step.id,
                    status="Error",
                    output_data=f"Circular dependency or unresolved dependencies detected."
                )
                raise RuntimeError("Circular dependency or unresolved dependencies detected.")

        print(f"🏁 All tasks completed. Total plan versions: {len(plan_versions)}")
        update_step(
            db=db,
            step_id=step.id,
            status="Completed",
            output_data={"outputs": outputs, "plan_versions": plan_versions}
        )
        return {"outputs": outputs, "plan_versions": plan_versions}



    # Plan Analyzer 
    def _build_analyze_prompts(self) -> dict:
        """
        Builds the LLM prompt templates for generating responses,
        """
        
        business_context = load_prompt("0.business_context")
        data_sources = load_prompt("1.data_sources_context")
        tools_planning = load_prompt("2.tools_planning")
        plan_update_prompt = business_context + data_sources + tools_planning+ load_prompt("2.plan_update")
        return ChatPromptTemplate.from_messages([
                ("system", plan_update_prompt)
            ])
    

    def _analyze_and_update_plan(self, question: str, plan: list, latest_output: Any, previous_outputs: dict) -> list:
        """
        Invokes the Analyzer LLM to interpret the latest tool output,
        reason about what to do next, and update the remaining plan accordingly.
        """

        latest_tool_output = json.dumps(latest_output, indent=2, ensure_ascii=False)
        full_tool_output = json.dumps(list(previous_outputs.keys()), indent=2)
        remaining_plan = json.dumps(plan, indent=2, ensure_ascii=False)        
        # Call the Analyzer LLM
        try:
            analyzer_prompt = self.analyze_prompt_plan.invoke({"question": question, 
                                                            "latest_tool_output": latest_tool_output, 
                                                            "full_tool_output": full_tool_output,
                                                            "remaining_plan": remaining_plan })
            response = self.base_llm.invoke(analyzer_prompt)
            text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            print(f"⚠️ LLM invocation failed: {e}")
            return plan  # Fallback: continue with current plan

        #  Validate JSON output
        try:
            updated_plan_structured = self.plan_structure_llm.invoke(text)
            updated_plan = updated_plan_structured["plan"]
            if not isinstance(updated_plan, list):
                raise ValueError("Analyzer must return a list of tasks.")
        except Exception as e:
            print(f"⚠️ Invalid plan format returned by analyzer: {e}")
            print(f"Raw LLM output:\n{text}")
            return plan
        print("🧩 Analyzer LLM produced an updated plan successfully.")
        return updated_plan