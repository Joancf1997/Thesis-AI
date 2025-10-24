import io
import os
import json
import uuid
import copy
import pandas as pd
from pydantic import Field
from dotenv import load_dotenv
from PIL import Image as PILImage
from typing_extensions import TypedDict
from typing import Dict, List, Optional, Any

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
        self.state: State = {"question": "", "plan": [], "validation": False, "outputs": {}, "response": ""}
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
        self.plan_update = self.planning_prompt + load_prompt('2.plan_update')
        self.response_prompt = self.business_context + load_prompt('4.response_stage')
        self.direct_response_prompt = self.business_context + self.data_sources + load_prompt('3.direct_response')
        self.task_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", self.planning_prompt),
            ("human", "Question: {question}")
        ])
        self.plan_update_prompt = ChatPromptTemplate.from_messages([
            ("system", self.plan_update)
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
          print(step)
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
        plan_versions = [copy.deepcopy(state["plan"])]  # Track all plan versions
        remaining = copy.deepcopy(state["plan"])

        # ===============================
        #  Type mapping per tool
        # ===============================
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

        # ===============================
        #  Utility functions
        # ===============================
        def extract_property(obj, prop_path):
            """Safely extract nested property (supports dot notation)."""
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

        def cast_arg(value, expected_type):
            """Safely cast arguments according to the expected type."""
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
                        raise ValueError(f"Dependency '{dep_task_id}' not yet available for argument '{k}'.")
                    dep_output = outputs[dep_task_id]

                    if prop:
                        dep_output = extract_property(dep_output, prop)

                    resolved[k] = dep_output
                else:
                    resolved[k] = v

                # Cast according to expected type
                expected_type = TOOL_ARG_TYPES.get(task["task"], {}).get(k)
                if expected_type:
                    resolved[k] = cast_arg(resolved[k], expected_type)

            return resolved

        # ===============================
        #  Main execution loop
        # ===============================
        while remaining:
            progress = False
            for task in remaining[:]:
                # Check dependencies
                if all(dep in outputs for dep in task.get("dep", [])):
                    args = resolve_args(task)
                    func = self.tools.TASK_FUNCS.get(task["task"])
                    if not func:
                        raise ValueError(f"Unknown task function: {task['task']}")

                    # Execute tool function
                    result = func(**args)
                    result_serializable = make_serializable(result)
                    outputs[task["id"]] = result_serializable
                    remaining.remove(task)
                    progress = True

                    print(f"✅ Executed: {task['task']} | ID: {task['id']}")

                    # -----------------------------------
                    #  ANALYSIS PHASE
                    # -----------------------------------
                    if task.get("analyze_answer", False):
                        target_prop = task.get("analyze_target_property")
                        if target_prop:
                            try:
                                target_value = extract_property(result_serializable, target_prop)
                            except Exception as e:
                                print(f"⚠️ Warning: Could not extract property '{target_prop}' for analysis: {e}")
                                target_value = result_serializable
                        else:
                            target_value = result_serializable

                        # 🔍 Call the Analyzer LLM
                        print(f"🧠 Triggering analysis for task '{task['id']}'...")
                        new_plan = self.analyze_and_update_plan(
                            question=state["question"],
                            plan=remaining,
                            latest_output=target_value,
                            previous_outputs=outputs
                        )

                        # ✅ Validate the new plan
                        validation_result = self.validate_plan({"plan": new_plan})
                        is_valid = validation_result.get("validation", False)

                        if is_valid:
                            plan_versions.append(copy.deepcopy(new_plan))
                            print(f"🔄 Plan updated → version {len(plan_versions)} (valid)")
                            remaining = copy.deepcopy(new_plan)
                        else:
                            print("⚠️ New plan from analyzer is invalid — keeping previous plan.")
                            plan_versions.append(copy.deepcopy(remaining))
                        break  # restart loop with updated plan

            if not progress:
                raise RuntimeError("Circular dependency or missing dependencies detected")

        # ===============================
        #  Final output
        # ===============================
        print(f"🏁 All tasks completed. Total plan versions: {len(plan_versions)}")

        # ===============================
        #  Visualization and Export
        # ===============================

        # Save all plan versions to JSON
        os.makedirs("plan_versions", exist_ok=True)
        for i, plan in enumerate(plan_versions, 1):
            with open(f"plan_versions/plan_v{i}.json", "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(plan_versions)} plan versions to /plan_versions/")

        # Optional: visualize the plan flow for each version
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            def visualize_plan(plan, version):
                G = nx.DiGraph()
                for task in plan:
                    G.add_node(task["id"], label=task["task"])
                    for dep in task.get("dep", []):
                        G.add_edge(dep, task["id"])

                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2500, font_size=9)
                nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes})
                plt.title(f"Plan Version {version}")
                plt.savefig(f"plan_versions/plan_v{version}.png")
                plt.close()

            for i, plan in enumerate(plan_versions, 1):
                visualize_plan(plan, i)

            print(f"🖼️ Generated dependency graphs for all plan versions.")
        except ImportError:
            print("⚠️ Skipping visualization (install networkx and matplotlib to enable).")

        return {"outputs": outputs, "plan_versions": plan_versions}
    

    def analyze_and_update_plan(self, question: str, plan: list, latest_output: Any, previous_outputs: dict) -> list:
        """
        Invokes the Analyzer LLM to interpret the latest tool output,
        reason about what to do next, and update the remaining plan accordingly.
        """

        latest_tool_output = json.dumps(latest_output, indent=2, ensure_ascii=False)
        full_tool_output = json.dumps(list(previous_outputs.keys()), indent=2)
        remaining_plan = json.dumps(plan, indent=2, ensure_ascii=False)        

        # Call the Analyzer LLM
        try:
            analyzer_prompt = self.plan_update_prompt.invoke({"question": question, 
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
    

    def direct_response(self, state: State):
      prompt = self.generate_direct_response_prompt.invoke({"question": state["question"]})
      response = self.base_llm.invoke(prompt)
      return {"response": response}

    def generate_response(self, state: State):
      prompt = self.generate_response_prompt.invoke({"question": state["question"], "plan": state["plan"], "tool_outputs": state["outputs"]})
      response = self.base_llm.invoke(prompt)
      return {"response": response}
      
    