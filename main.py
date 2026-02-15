from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Initialize Chroma
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="ofac_chunks",
    embedding_function=openai_ef
)

# Load chunks only once
if collection.count() == 0:
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk["content"])
        metadatas.append({
            "source": chunk["source"],
            "document_type": chunk.get("document_type"),
            "status": chunk.get("status"),
            "date": chunk.get("date")
        })
        ids.append(f"id_{i}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "OFAC RAG backend running"}

@app.post("/ask")
def ask_question(data: Question):
    results = collection.query(
        query_texts=[data.question],
        n_results=5
    )

    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a legal compliance assistant specialized in OFAC sanctions. Answer only using the provided context. If the answer is not in the context, say you cannot determine it."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{data.question}"
            }
        ]
    )

    return {"answer": response.choices[0].message.content}
