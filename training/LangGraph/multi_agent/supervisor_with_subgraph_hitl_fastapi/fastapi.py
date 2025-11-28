from fastapi import FastAPI
from langgraph.types import Command
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

class WriteEssayRequest:
    thread_id: str = Field(default_factory= lambda: str(datetime.now()))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/essay/write")
def write_essay(thread_id: str, response: str = None):
    thread_config = {"configurable": {"thread_id": thread_id}}