from fastapi import FastAPI
from pydantic import BaseModel
from Assistant.ARDIChat import ChatAssistant
import uvicorn

from fastapi.middleware.cors import CORSMiddleware  # ✅ Add this

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

@app.post("/chat/ask")
def chat_endpoint(question: Question):
    response = chat.ask(question.question)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
