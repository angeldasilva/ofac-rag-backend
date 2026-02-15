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

      metadata = {
        "filename": str(chunk.get("filename", "")),
        "document_type": str(chunk.get("document_type", "")),
        "date": str(chunk.get("date", "")),
        "jurisdiction": str(chunk.get("jurisdiction", ""))
      }

      metadatas.append(metadata)
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
        n_results=8
    )

    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a legal compliance analyst specialized in OFAC sanctions. Use the provided context as the primary authoritative source. You may apply legal reasoning and interpret regulatory provisions logically. If relevant provisions exist in the context, analyze them and explain how they apply. If the context is insufficient, you may provide general regulatory reasoning but clearly state when you are extrapolating. Do not fabricate specific license permissions that are not supported by the context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{data.question}"
            }
        ]
    )

    return {"answer": response.choices[0].message.content}
