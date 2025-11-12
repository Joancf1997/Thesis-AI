import uuid
from sqlalchemy.orm import Session
from models.thread import Thread

# Optional: global user uuid (static user)
DEFAULT_USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")

def create_new_thread(
    db: Session,
    name: str = "ARDI - Assistant",       
    user_id: uuid.UUID = DEFAULT_USER_ID
):
    new_session = Thread(name=name, user_id=user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session
