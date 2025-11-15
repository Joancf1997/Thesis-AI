from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime

# ----------------------------------------
# MESSAGE (User or Agent message)
# ----------------------------------------
class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    response_to = Column(UUID(as_uuid=True))

    created_at = Column(DateTime, default=datetime.utcnow)

    thread = relationship("Thread", back_populates="messages")

    # A message may trigger a Run (agent workflow)
    run = relationship("Run", back_populates="message", uselist=False)
