# db/create_db.py
import os
import sys

# Get the project root (one level above /db)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db.base import Base, engine
from models.user import User
from models.thread import Thread
from models.run import Run
from models.step import Step
from models.message import Message
from models.toolCall import ToolCall
from models.datasetEvaluation import DatasetEvaluation

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Done.")
