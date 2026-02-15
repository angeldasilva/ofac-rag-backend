from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "OFAC RAG backend running"}

@app.post("/ask")
def ask_question(data: Question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # luego podemos cambiar a gpt-4o
        messages=[
            {"role": "system", "content": "You are a legal compliance assistant specialized in OFAC sanctions."},
            {"role": "user", "content": data.question}
        ]
    )

    return {"answer": response.choices[0].message.content}
