import uuid
import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from API.SystemAPI import load_user_segments, load_user_segments_detail
from Assistant.ARDIChat import ChatAssistant
from crud.thread import create_new_thread, load_threads, load_thread_messages, update_thread_name, remove_thread
from crud.users import load_users, create_user
from crud.login import login_user
from db.session import get_db
from fastapi.middleware.cors import CORSMiddleware 

chat = ChatAssistant()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Agent Endpoints 
# ===============================
# Create a new user session 
class CreateThreadRequest(BaseModel):
    user_id: uuid.UUID
    name: str = "ARDI - Assistant"
@app.post("/chat/newThread")
def create_thread(req: CreateThreadRequest, db: Session = Depends(get_db)):
    new_session = create_new_thread(
        db=db,
        user_id=req.user_id,
        name=req.name
    )
    return {
        "thread": {
            "id": str(new_session.id),
            "name": new_session.name,
            "user_id": str(new_session.user_id),
            "started_at": new_session.started_at,
        }
    }



# Ask the agent a question
class Question(BaseModel):
    question: str
    thread_id: str
@app.post("/chat/ask")
def chat_endpoint(question: Question, db: Session = Depends(get_db)):
    response = chat.ask(db, question)
    return {"response": response}


@app.get("/dataset_evaluation")
def dataset_evaluation(db: Session = Depends(get_db)):
    response = chat.evaluate_dataset(db)
    return {"evaluarion": response}

# ===============================
# users 
# ===============================
@app.get("/users")
def list_users(db: Session = Depends(get_db)):
    users = load_users(db)
    return {"users": users}


class CreateUserRequest(BaseModel):
    username: str
    password: str
@app.post("/users/create")
def create_user_endpoint(
    req: CreateUserRequest,
    db: Session = Depends(get_db)
):
    user = create_user(db, req)
    return {"user": user}



class LoginRequest(BaseModel):
    username: str
    password: str
@app.post("/auth/login")
def login_endpoint(
    req: LoginRequest, 
    db: Session = Depends(get_db)
):
    result = login_user(db, req)
    return {"login": result}





# ===============================
# System Endpoints 
# ===============================
# Get all user segments
@app.get("/UserSegments")
def get_user_segments():
    response = load_user_segments()
    return {"segments": response}


# Get details for one segment by ID
@app.get("/UserSegment/{id}")
def get_user_segment_detail(id: str):
    response = load_user_segments_detail(id)
    return {"segment_detail": response}


from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Any


class AskRequest(BaseModel):
    question: str
    thread_id: UUID
    user_id: UUID


class NewThreadRequest(BaseModel):
    user_id: UUID



class MessageResponse(BaseModel):
    role: str
    content: str
    outputs: Optional[Any]
    timestamp: str

# Get User threads
@app.get("/threads")
def get_user_threads(user_id: UUID, db: Session = Depends(get_db)):
    return load_threads(db, user_id)

@app.get("/chat/history/{thread_id}")
def get_history(
    thread_id: UUID,
    user_id: UUID,
    db: Session = Depends(get_db)
):
    return load_thread_messages(db, thread_id, user_id)



class RenameThreadRequest(BaseModel):
    name: str
    user_id: UUID
@app.put("/chat/thread/{thread_id}/rename")
def rename_thread(
    thread_id: UUID,
    req: RenameThreadRequest,
    db: Session = Depends(get_db)
):
    thread = update_thread_name(
        db=db,
        thread_id=thread_id,
        user_id=req.user_id,
        new_name=req.name
    )

    return {
        "thread": {
            "id": str(thread.id),
            "name": thread.name,
            "user_id": str(thread.user_id),
            "started_at": thread.started_at,
            "ended_at": thread.ended_at
        }
    }

class DeleteThreadRequest(BaseModel):
    user_id: UUID
@app.delete("/chat/thread/{thread_id}")
def delete_thread(
    thread_id: UUID,
    req: DeleteThreadRequest,
    db: Session = Depends(get_db)
):
    deleted = remove_thread(
        db=db,
        thread_id=thread_id,
        user_id=req.user_id
    )

    return {
        "deleted": True,
        "thread_id": str(deleted.id)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
