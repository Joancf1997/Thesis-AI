import uvicorn
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from API.SystemAPI import load_user_segments, load_user_segments_detail
from Assistant.ARDIChat import ChatAssistant
from crud.thread import create_new_thread
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
@app.post("/new_session")
def create_session(db: Session = Depends(get_db)):
    new_session = create_new_thread(db)
    return {
        "session_id": str(new_session.id),
        "name": new_session.name,
        "user_id": str(new_session.user_id),
        "started_at": new_session.started_at,
    }


# Ask the agent a question
class Question(BaseModel):
    question: str
    thread_id: str
@app.post("/chat/ask")
def chat_endpoint(question: Question, db: Session = Depends(get_db)):
    response = chat.ask(db, question)
    return {"response": response}





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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
