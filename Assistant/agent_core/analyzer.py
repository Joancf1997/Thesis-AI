import json
from typing import Any
from utils.prompt_loader import load_prompt 
from langchain_core.prompts import ChatPromptTemplate

class analyzer():
    def __init__(self):
        self.plan_update_prompt = self._build_response_prompts()

    def _build_response_prompts(self) -> dict:
        """
        Builds the LLM prompt templates for generating responses,
        """
        
        business_context = load_prompt("0.business_context")
        data_sources = load_prompt("1.data_sources_context")
        tools_planning = load_prompt("2.tools_planning")
        plan_update_prompt = business_context + data_sources + tools_planning+ load_prompt("4.plan_update")
        return ChatPromptTemplate.from_messages([
                ("system", plan_update_prompt)
            ])

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
