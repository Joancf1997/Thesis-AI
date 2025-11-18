from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime

# ----------------------------------------
# TOOL CALL (Tool usage inside a step)
# ----------------------------------------
class ToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    step_id = Column(UUID(as_uuid=True), ForeignKey("steps.id", ondelete="CASCADE"))
    tool_name = Column(String)
    input = Column(JSON)
    output = Column(JSON)
    status = Column(String)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    meta = Column(JSON)

    step = relationship("Step", back_populates="tool_calls")