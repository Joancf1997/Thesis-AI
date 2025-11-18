from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime

# ----------------------------------------
# STEP (Specific tool execution)
# ----------------------------------------
class Step(Base):
    __tablename__ = "steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"))
    name = Column(String)
    input = Column(JSON)
    output = Column(JSON)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="steps")

    tool_calls = relationship(
        "ToolCall",
        back_populates="step",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
