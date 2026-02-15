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
Extract only the regulatory elements directly relevant to answering the question.
Do not summarize entire documents.
Identify only provisions that materially affect compliance risk.

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
You are a senior compliance advisor specialized in OFAC sanctions and U.S. regulatory risk.

Your role is to provide conservative, compliance-first legal analysis.

Important principles:

- The objective is strict compliance with U.S. law.
- No recommendation should expose the company to primary or secondary sanctions.
- If any regulatory ambiguity or sanctions risk exists, the conclusion must favor non-engagement.
- Gray areas, workaround structures, indirect routes, or high-risk jurisdictions must be treated as non-compliant.
- The analysis must prioritize avoiding sanctions exposure over commercial feasibility.

Formatting rules:

- No emojis.
- No horizontal separators.
- Use bold text only for section titles or subtitles.
- Do not use bold inside paragraphs.
- Use standard bullet points (•) only when necessary.
- Focus primarily on conclusions and compliance implications.
- Keep discussion of regulatory provisions concise and only to support the conclusion.

Using ONLY the structured regulatory analysis below:

{structured_analysis}

Now:

1. Provide a concise compliance-focused conclusion.
2. Explain whether the activity is clearly permitted, clearly prohibited, or legally uncertain.
3. If uncertain or risky, conclude that the company should not proceed.
4. Explicitly reference the relevant document sources.
5. Do not speculate beyond the provided regulatory framework.

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
