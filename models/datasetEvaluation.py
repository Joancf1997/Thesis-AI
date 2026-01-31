from sqlalchemy import Column, String, Integer, ForeignKey, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from db.base import Base
import uuid
from datetime import datetime


class DatasetEvaluation(Base):
    __tablename__ = "dataset_evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    question = Column(String, nullable=False)
    labels = Column(JSON, nullable=False)       
    actual_tools = Column(JSON, nullable=False) 
    matched = Column(Integer, nullable=False)
    total_labels = Column(Integer, nullable=False)
    match_ratio = Column(Float, nullable=False)      
    match_ratio_str = Column(String, nullable=False) 
    created_at = Column(DateTime, default=datetime.utcnow)
    thread = relationship("Thread", back_populates="evaluations")
