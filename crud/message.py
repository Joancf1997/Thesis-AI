# crud/message.py

import uuid
from sqlalchemy.orm import Session
from models.message import Message

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


def create_assistant_message(db: Session, thread_id: uuid.UUID, content: str) -> Message:
    """
    Inserts a Assistant message (role='Assistant') into the Message table
    linked to a Thread.
    """
    message = Message(
        thread_id=thread_id,
        role="Assistant",
        content=content
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message
