import json
from utils.prompt_loader import load_prompt 
from langchain_core.prompts import ChatPromptTemplate

class responder():
    def __init__(self, llm): 
        self.generate_response_prompt, self.direct_response_prompt = self._build_response_prompts()
        self.llm = llm

    def _build_response_prompts(self) -> dict:
        """
        Builds the LLM prompt templates for generating responses,
        both when tools are used (generate_response) and when not (direct_response).
        """
        business_context = load_prompt("0.business_context")
        data_sources = load_prompt("1.data_sources_context")
        direct_response_prompt = load_prompt("3.direct_response")
        response_prompt = business_context + load_prompt("4.response_stage")
        direct_response_prompt = business_context + data_sources + load_prompt("3.direct_response")
        return {
            "generate_response_prompt": ChatPromptTemplate.from_messages([
                ("system", response_prompt)
            ]),
            "direct_response_prompt": ChatPromptTemplate.from_messages([
                ("system", direct_response_prompt)
            ])
        }

    def generate_response(self, state):
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
            response = self.llm.invoke(prompt)
            return {"response": response}
        except Exception as e:
            print(f"⚠️ Error generating response: {e}")
            return {"response": f"An error occurred while generating the final response: {str(e)}"}

    def direct_response(self, state):
        """
        Handles questions that do not require tool execution.
        Produces a direct natural-language answer using only the context.
        """
        try:
            prompt = self.direct_response_prompt.invoke({"question": state["question"]})
            response = self.llm.invoke(prompt)
            return {"response": response}
        except Exception as e:
            print(f"⚠️ Error generating direct response: {e}")
            return {"response": f"An error occurred while generating the direct response: {str(e)}"}
