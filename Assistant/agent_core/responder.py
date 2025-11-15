import json
from typing import Dict
from sqlalchemy.orm import Session
from utils.utils import load_prompt 
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from crud.step import create_step, update_step

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

    def generate_response(self, db: Session, state: State):
        """
        Combines the user question, plan, and tool outputs
        into a final natural-language answer.
        """
        step = create_step(
            db=db,
            run_id=state["run_id"],
            name="Response Generation",
            input_data={
                "question": state.get("question", ""),
                "plan": json.dumps(state.get("plan", {}), indent=2, ensure_ascii=False),
                "tool_outputs": json.dumps(state.get("outputs", {}), indent=2, ensure_ascii=False)
            }
        )

        try:
            prompt = self.generate_response_prompt.invoke({
                "question": state.get("question", ""),
                "plan": json.dumps(state.get("plan", {}), indent=2, ensure_ascii=False),
                "tool_outputs": json.dumps(state.get("outputs", {}), indent=2, ensure_ascii=False)
            })

            response = self.base_llm.invoke(prompt)

            update_step(
                db=db,
                step_id=step.id,
                status="Completed",
                output_data={"response": response.content}
            )

            return {"response": response.content}

        except Exception as e:
            print(f"⚠️ Error generating final response: {e}")
            update_step(
                db=db,
                step_id=step.id,
                status="Error",
                output_data={"error": str(e)}
            )
            # Re-raise to stop workflow
            raise RuntimeError(f"Response generation failed: {e}") from e


    def direct_response(self, db: Session, state: State):
        """
        Handles questions that do not require tool execution.
        Produces a direct natural-language answer using only the context.
        """
        step = create_step(
            db=db,
            run_id=state["run_id"],
            name="Direct Response",
            input_data=state.get("question", "")
        )

        try:
            prompt = self.direct_response_prompt.invoke({
                "question": state.get("question", "")
            })

            response = self.base_llm.invoke(prompt)

            update_step(
                db=db,
                step_id=step.id,
                status="Completed",
                output_data={"response": response.content}
            )

            return {"response": response.content}

        except Exception as e:
            print(f"⚠️ Error generating direct response: {e}")
            update_step(
                db=db,
                step_id=step.id,
                status="Error",
                output_data={"error": str(e)}
            )
            # Re-raise to stop workflow
            raise RuntimeError(f"Direct response generation failed: {e}") from e

