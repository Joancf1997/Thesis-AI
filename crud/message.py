# crud/message.py

import uuid
from sqlalchemy.orm import Session
from models.message import Message
from .run import update_run_message_id


def create_human_message(db: Session, thread_id: uuid.UUID, content: str) -> Message:
    """
    Inserts a HUMAN message (role='user') into the Message table
    linked to a Thread.
    """
    message = Message(
        thread_id=thread_id,
        role="user",
        content=content
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


def create_assistant_message(
        db: Session, 
        thread_id: uuid.UUID, 
        content: str, 
        resp_msg_id: uuid.UUID
    ) -> Message:
    """
    Inserts a Assistant message (role='Assistant') into the Message table
    linked to a Thread.
    """
    # Create assistant message 
    message = Message(
        thread_id=thread_id,
        role="assistant",
        content=content,
        response_to=resp_msg_id
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    # Update the step to refer to this agent message
    update_run_message_id(db, message_id=resp_msg_id, new_message_id=message.id)

    return message
