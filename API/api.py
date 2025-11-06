import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from API.SystemAPI import load_user_segments, load_user_segments_detail
from Assistant.ARDIChat import ChatAssistant

from fastapi.middleware.cors import CORSMiddleware 

chat = ChatAssistant()
app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str


# ✅ Assistant API 
@app.post("/chat/ask")
def chat_endpoint(question: Question):
    response = chat.ask(question.question)
    return {"response": response}


# ✅ System API: Get all user segments
@app.get("/UserSegments")
def get_user_segments():
    response = load_user_segments()
    return {"segments": response}


# ✅ System API: Get details for one segment by ID
@app.get("/UserSegment/{id}")
def get_user_segment_detail(id: str):
    response = load_user_segments_detail(id)
    return {"segment_detail": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
