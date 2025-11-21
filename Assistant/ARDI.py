import os
import uuid
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain.chat_models import init_chat_model
import pandas as pd
from collections import defaultdict

# Internal imports
from utils.utils import Settings
from .agent_core.workflow import Workflow
from .agent_core.planner import Plan  
from db.insert_dataset import DatasetEntry
from crud.message import create_human_message, create_assistant_message
from models.thread import Thread
from models.message import Message
from models.run import Run
from models.step import Step
from models.toolCall import ToolCall
from sqlalchemy.orm import aliased


from models.datasetEvaluation import DatasetEvaluation
import psycopg2
import psycopg2.extras

from crud.run import create_run, end_run

# ==========================================================
#  ARDI AGENT
# ==========================================================
class Agent:
    """
    The ARDI Agent orchestrates the LLM-powered analytical workflow.
    It handles initialization, configuration, and user question resolution.
    """

    def __init__(self, settings: Settings):
        load_dotenv()

        # Session & config
        self.settings = settings
        self.llm_config = settings.llm
        print(self.llm_config )

        # Initialize everything
        self._init_llms()
        self._init_workflow()
        print("ARDI Agent ready...")

    def _init_llms(self):
        """Initialize both the base LLM and structured-output LLM."""
        self.base_llm = init_chat_model(
            model=self.llm_config.model_name,
            model_provider=self.llm_config.provider,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
        )

        # Structured LLM for JSON plan schema
        self.plan_structure_llm = self.base_llm.with_structured_output(Plan)

    # ------------------------------------------------------
    #  WORKFLOW INITIALIZATION
    # ------------------------------------------------------
    def _init_workflow(self):
        """Build the LangGraph workflow with all nodes connected."""
        self.workflow = Workflow(
            base_llm=self.base_llm,
            plan_structure_llm=self.plan_structure_llm
        )
        self.request = self.workflow.graph

    # ------------------------------------------------------
    #  MAIN ENTRYPOINT - Capture User question
    # ------------------------------------------------------
    def ask(self, db: Session, message_obj):
        """
        Executes the full agent pipeline on a user question.
        """

        thread_id = message_obj.thread_id
        message_content = message_obj.content
        session_config = {"configurable": {"thread_id": str(thread_id)}}
        print(f"User question: {message_content}\n")
        execution_result = {}

        run = create_run(db, message_obj.id)
        self.workflow.set_db(db)
        self.workflow.failed = False
        state = {
            "question": message_content, 
            "thread_id": thread_id,
            "run_id": run.id
        }
        for step in self.request.stream(
            state,
            session_config,
            stream_mode="updates"
        ):
            print(f"📍 Step update: {step}")
            key = list(step.keys())[0]         
            execution_result[key] = step[key]
        
        end_run(db, run.id)
        print("\n✅ Agent pipeline finished successfully!\n")
        return execution_result 
    

    def process_dataset_entries(
            self, 
            db: Session,
            user_id: uuid.UUID,
            name: str = "Dataset Evaluation"
        ):
            # Create the new evaluation thread
            new_session = Thread(name=name, user_id=user_id)
            db.add(new_session)
            db.commit()
            db.refresh(new_session)

            thread_id = new_session.id

            # Fetch dataset entries
            entries = db.query(DatasetEntry).all()
            print(f"📦 Found {len(entries)} dataset entries to process.")

            # --- MAIN EXECUTION LOOP ---
            for i, entry in enumerate(entries, start=1):
                print(f"\n🚀 Processing entry {i}/{len(entries)} | ID: {entry.id}")
                print(entry.user_query)

                # Create human message
                human_msg = create_human_message(
                    db, thread_id=thread_id, content=entry.user_query
                )
                print(human_msg.content)

                execution = self.ask(db, human_msg)

                # Extract assistant response
                resp_obj = execution.get("direct_response") or execution.get("generate_response")
                response_text = resp_obj["response"] if resp_obj else None

                # Save assistant message
                create_assistant_message(
                    db, thread_id=thread_id, content=response_text, resp_msg_id=human_msg.id
                )
                print(response_text)


            def evaluate_tool_usage(rows):
                grouped = defaultdict(lambda: {"labels": set(), "actual": set()})

                for row in rows:
                    q = row.question   # FIXED

                    if row.tools_used:
                        grouped[q]["labels"] = set(row.tools_used)

                    if row.tool_name is not None:
                        grouped[q]["actual"].add(row.tool_name)

                results = []

                for question, data in grouped.items():
                    labels = data["labels"]
                    actual = data["actual"]

                    matched = len(actual.intersection(labels))
                    total_labels = len(labels)

                    ratio = matched / total_labels if total_labels > 0 else 0
                    ratio_str = f"{matched}/{total_labels}" if total_labels > 0 else "0/0"

                    results.append({
                        "question": question,
                        "labels": sorted(labels),
                        "actual_tools": sorted(actual),
                        "matched": matched,
                        "total_labels": total_labels,
                        "match_ratio": ratio,
                        "match_ratio_str": ratio_str
                    })

                results.sort(key=lambda x: x["question"])
                return results


            MessageAlias = aliased(Message)

            rows = (
                db.query(
                    Message.content.label("question"),
                    ToolCall.tool_name.label("tool_name"),
                    DatasetEntry.tools_used.label("tools_used")
                )
                # Thread → Message
                .join(Thread, Message.thread_id == Thread.id)
                # Message → user_msg
                .outerjoin(MessageAlias, MessageAlias.response_to == Message.id)
                # user_msg → Run
                .outerjoin(Run, Run.message_id == MessageAlias.id)
                # Run → Step
                .outerjoin(
                    Step,
                    (Step.run_id == Run.id) & (Step.name == "Plan Execution")
                )
                # Step → ToolCall
                .outerjoin(ToolCall, ToolCall.step_id == Step.id)
                # Message.content → DatasetEntry.user_query
                .join(DatasetEntry, DatasetEntry.user_query == Message.content)
                .filter(Thread.id == thread_id)
                .order_by(Message.content.asc())
                .all()
            )

            # ---------------------------------------------------------
            # Process evaluation results
            # ---------------------------------------------------------
            print(rows)
            results = evaluate_tool_usage(rows)
            print(results)
            df = pd.DataFrame(results)
            print(df)

            for _, row in df.iterrows():
                print(row)
                evaluation = DatasetEvaluation(
                    thread_id=thread_id,
                    question=row["question"],
                    labels=row["labels"],
                    actual_tools=row["actual_tools"],
                    matched=row["matched"],
                    total_labels=row["total_labels"],
                    match_ratio=row["match_ratio"],
                    match_ratio_str=row["match_ratio_str"]
                )
                db.add(evaluation)

            db.commit()
            return