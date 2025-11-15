import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class DatasetEntry(Base):
    __tablename__ = "dataset_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(String, nullable=False)
    task_plan = Column(String, nullable=False)   # stored as JSON string
    tools_used = Column(JSON, nullable=False)    # list of strings
    focus = Column(String, nullable=False)
    complexity = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<DatasetEntry(id={self.id}, query='{self.user_query[:40]}...', complexity={self.complexity})>"



def insert_dataset_from_json(json_path: str, db_url: str = "postgresql://joseandres:@localhost/ardi_dev"):
    """
    Reads dataset entries from a JSON file and inserts them into the database.

    JSON file format:
    [
      {
        "user_query": "string",
        "task_plan": "string (JSON)",
        "tools_used": ["list", "of", "tools"],
        "focus": "string",
        "complexity": int
      },
      ...
    ]
    """
    # Load dataset
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read JSON file: {e}")
        return

    # Setup engine and session
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        entries = []
        for item in dataset:
            entry = DatasetEntry(
                user_query=item["user_query"],
                task_plan=item["task_plan"],
                tools_used=item["tools_used"],
                focus=item["focus"],
                complexity=item["complexity"]
            )
            entries.append(entry)
        session.add_all(entries)
        session.commit()
        print(f"✅ Successfully inserted {len(entries)} entries into the database.")
    except Exception as e:
        session.rollback()
        print(f"❌ Error inserting dataset: {e}")
    finally:
        session.close()


# insert_dataset_from_json("/Users/joseandres/Documents/Thesis/Thesis-AI/datasets/evaluation_dataset.json")