import json
from typing import Dict
from typing_extensions import TypedDict
from utils.utils import load_prompt 
from langchain_core.prompts import ChatPromptTemplate

class State(TypedDict):
    question: str
    plan: Dict
    validation: bool
    outputs: Dict
    response: str

class Responder():
    def __init__(self, base_llm): 
        self.generate_response_prompt, self.direct_response_prompt = self._build_response_prompts()
        self.base_llm = base_llm

    def _build_response_prompts(self) -> dict:
        """
        Builds the LLM prompt templates for generating responses,
        both when tools are used (generate_response) and when not (direct_response).
        """
        business_context = load_prompt("0.business_context")
        data_sources = load_prompt("1.data_sources_context")
        response_prompt = business_context + load_prompt("4.response_stage")
        direct_response = business_context + data_sources + load_prompt("3.direct_response")
        return[
            ChatPromptTemplate.from_messages([
                ("system", response_prompt)
            ]),
            ChatPromptTemplate.from_messages([
                ("system", direct_response)
            ])
        ]

    def generate_response(self, state: State):
        """
        Combines the user question, plan, and tool outputs
        into a final natural-language answer.
        """
        try:
            prompt = self.generate_response_prompt.invoke({
                "question": state["question"],
                "plan": json.dumps(state["plan"], indent=2, ensure_ascii=False),
                "tool_outputs": json.dumps(state["outputs"], indent=2, ensure_ascii=False)
            })
            response = self.base_llm.invoke(prompt)
            return {"response": response}
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return {"response": f"An error occurred while generating the final response: {str(e)}"}

    def direct_response(self, state: State):
        """
        Handles questions that do not require tool execution.
        Produces a direct natural-language answer using only the context.
        """
        try:
            prompt = self.direct_response_prompt.invoke({"question": state["question"]})
            response = self.base_llm.invoke(prompt)
            return {"response": response}
        except Exception as e:
            print(f"⚠️ Error generating direct response: {e}")
            return {"response": f"An error occurred while generating the direct response: {str(e)}"}
