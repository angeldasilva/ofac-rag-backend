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

    # --- STEP 1: RETRIEVAL ---
    results = collection.query(
        query_texts=[data.question],
        n_results=10
    )

    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    context_blocks = []
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        context_blocks.append(
            f"[Source: {meta.get('filename','')} | "
            f"Type: {meta.get('document_type','')} | "
            f"Date: {meta.get('date','')}]\n{doc}"
        )

    context = "\n\n".join(context_blocks)

    # --- STEP 2: STRUCTURED EXTRACTION ---
    extraction_prompt = f"""
You are a senior OFAC sanctions analyst.

From the provided regulatory context:

1. Identify relevant regulatory provisions.
2. Summarize what they permit or prohibit.
3. Identify conditions, limitations, or expiration clauses.
4. Extract only legally relevant elements.

Context:
{context}

Question:
{data.question}
"""

    extraction = client.chat.completions.create(
        model="gpt-5.2-chat-latest",
        messages=[
            {"role": "system", "content": "Extract structured regulatory elements only."},
            {"role": "user", "content": extraction_prompt}
        ]
    )

    structured_analysis = extraction.choices[0].message.content

    # --- STEP 3: LEGAL APPLICATION ---
    final_prompt = f"""
You are a senior compliance advisor specialized in OFAC sanctions.

Using ONLY the structured regulatory analysis below:

{structured_analysis}

Now:

1. Apply the regulatory provisions to the specific question.
2. Provide a clear legal conclusion.
3. State conditions or uncertainties.
4. Cite the document sources explicitly.
5. Do not invent permissions not supported by the analysis.

Question:
{data.question}
"""

    final_response = client.chat.completions.create(
        model="gpt-5.2-chat-latest",
        messages=[
            {"role": "system", "content": "Provide structured legal analysis with explicit citations."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return {
        "answer": final_response.choices[0].message.content
    }
