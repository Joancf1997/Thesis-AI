from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime

# ----------------------------------------
# RUN (Agent workflow)
# ----------------------------------------
class Run(Base):
    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"))
    status = Column(String)              # "queued" | "running" | "completed" | "failed"
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    message = relationship("Message", back_populates="run")
    steps = relationship("Step", back_populates="run")
