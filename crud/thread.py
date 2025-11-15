import json
import uuid
from models.thread import Thread
from models.message import Message
from models.step import Step
from models.run import Run
from fastapi import HTTPException
from sqlalchemy.orm import Session


def create_new_thread(
    db: Session,
    user_id: uuid.UUID,
    name: str = "ARDI - Assistant"
):
    new_session = Thread(name=name, user_id=user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def load_threads(
    db: Session,
    user_id: uuid.UUID
):
    threads = (
        db.query(Thread)
        .filter(Thread.user_id == user_id)
        .order_by(Thread.started_at.desc())
        .all()
    )
    return threads


def load_thread_messages(
    db: Session,
    thread_id: uuid.UUID,
    user_id: uuid.UUID
):
    # Validate thread ownership
    thread = (
        db.query(Thread)
        .filter(Thread.id == thread_id, Thread.user_id == user_id)
        .first()
    )

    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found or not owned by this user")

    # Load messages ordered by created_at ASC
    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    enriched_messages = []

    for msg in messages:
        outputs_ref = []
        run = (
            db.query(Run)
            .filter(Run.message_id == msg.id)
            .first()
        )

        if run:
            plan_execution = (
                db.query(Step)
                .filter(Step.run_id == run.id, Step.name == "Plan Execution")
                .order_by(Step.created_at.desc())
                .first()
            )   
            if plan_execution: 
                print("Plan")
                try:
                    print(plan_execution.output)
                    output_dict = plan_execution.output or {}
                    outputs = output_dict.get("outputs", [])
                    for output in outputs:
                        outputs_ref.append(outputs[output])
                except (ValueError, TypeError):
                    outputs = []



        enriched_messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "response_to": msg.response_to,
            "timestamp": int(msg.created_at.timestamp() * 1000),
            "outputs": outputs_ref
        })

    return enriched_messages


def update_thread_name(
    db: Session,
    thread_id: uuid.UUID,
    user_id: uuid.UUID,
    new_name: str
):
    thread = (
        db.query(Thread)
        .filter(Thread.id == thread_id, Thread.user_id == user_id)
        .first()
    )

    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found or not owned by this user")

    thread.name = new_name
    db.commit()
    db.refresh(thread)

    return thread