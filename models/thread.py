from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime

# ----------------------------------------
# THREAD  (Conversation context)
# ----------------------------------------
class Thread(Base):
    __tablename__ = "threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    name = Column(String, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    messages = relationship("Message", back_populates="thread")
