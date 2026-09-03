from __future__ import annotations

import asyncio
import io
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"

INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

load_dotenv(ROOT / ".env", override=True)


# ============================================================
# CONFIG
# ============================================================

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY",
    ""
).strip()

SARVAM_STT_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v3"
).strip()

SARVAM_CHAT_MODEL = os.getenv(
    "SARVAM_CHAT_MODEL",
    "sarvam-105b-conversations"
).strip()

GROUNDING_THRESHOLD = float(
    os.getenv(
        "GROUNDING_THRESHOLD",
        "0.55"
    )
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

SARVAM_STT_URL = (
    "https://api.sarvam.ai/"
    "speech-to-text"
)

SARVAM_CHAT_URL = (
    "https://api.sarvam.ai/"
    "v1/chat/completions"
)


# ============================================================
# STATE
# ============================================================

class State:
    index = None
    chunks = []
    model = None


state = State()


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print()
    print("=" * 60)
    print(" HH GOA 2026 - SARVAM VOICE RAG")
    print("=" * 60)

    if (
        INDEX_PATH.exists()
        and
        CHUNKS_PATH.exists()
    ):

        state.index = faiss.read_index(
            str(INDEX_PATH)
        )

        with open(
            CHUNKS_PATH,
            "rb"
        ) as file:

            state.chunks = pickle.load(
                file
            )

        print(
            "Loading multilingual embedding model..."
        )

        state.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        state.model.encode(
            ["hello"],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        print(
            f"✅ RAG READY: "
            f"{state.index.ntotal} vectors / "
            f"{len(state.chunks)} chunks"
        )

    else:

        print("⚠️ RAG files missing.")
        print("Run: python backend\\prepare.py")

    print(
        "Sarvam:",
        "✅ configured"
        if SARVAM_API_KEY
        else "❌ missing"
    )

    print(
        "Chat model:",
        SARVAM_CHAT_MODEL
    )

    print("=" * 60)
    print()

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="HH Goa 2026 Voice RAG",
    lifespan=lifespan
)


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.get("/")
def welcome():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


@app.get("/app")
def assistant():

    return FileResponse(
        FRONTEND_DIR / "app.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",

        "vectors": (
            state.index.ntotal
            if state.index is not None
            else 0
        ),

        "chunks": len(state.chunks),

        "rag_ready": (
            state.index is not None
            and state.model is not None
        ),

        "sarvam_ready": bool(
            SARVAM_API_KEY
        )
    }


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve(
    question: str,
    top_k: int = 5
):

    if (
        state.index is None
        or
        state.model is None
    ):

        return [], 0.0, 0.0

    start = time.perf_counter()

    embedding = state.model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    count = min(
        top_k,
        state.index.ntotal
    )

    scores, ids = state.index.search(
        embedding,
        count
    )

    retrieval_ms = (
        time.perf_counter()
        - start
    ) * 1000

    hits = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        if idx < 0:
            continue

        if idx >= len(
            state.chunks
        ):
            continue

        item = state.chunks[idx]

        if isinstance(
            item,
            dict
        ):

            text = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

        else:

            text = str(
                item
            ).strip()

            item = {
                "text": text,
                "source": "MSMARCO-XI"
            }

        if not text:
            continue

        hits.append({
            **item,

            "text": text,

            "score": float(
                score
            )
        })

    confidence = (
        hits[0]["score"]
        if hits
        else 0.0
    )

    return (
        hits,
        confidence,
        retrieval_ms
    )


# ============================================================
# SARVAM STT
# ============================================================

def sarvam_transcribe(
    audio_bytes: bytes,
    filename: str,
    mime_type: str
):

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is missing."
        )

    try:

        response = requests.post(
            SARVAM_STT_URL,

            headers={
                "api-subscription-key":
                    SARVAM_API_KEY
            },

            files={
                "file": (
                    filename,
                    io.BytesIO(
                        audio_bytes
                    ),
                    mime_type
                )
            },

            data={
                "model":
                    SARVAM_STT_MODEL,

                "mode":
                    "transcribe"
            },

            timeout=60
        )

    except requests.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect to "
                "Sarvam Speech-to-Text."
            )
        ) from error

    if not response.ok:

        print()
        print(
            "❌ SARVAM STT ERROR:",
            response.status_code
        )

        print(
            response.text[:1200]
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam STT failed: "
                f"{response.status_code}"
            )
        )

    data = response.json()

    transcript = str(
        data.get(
            "transcript",
            ""
        )
    ).strip()

    language = (
        data.get(
            "language_code"
        )
        or
        "unknown"
    )

    if not transcript:

        raise HTTPException(
            status_code=422,
            detail=(
                "No speech was detected. "
                "Please speak again."
            )
        )

    print()
    print("=" * 60)
    print(
        "🎤 QUESTION:",
        transcript
    )
    print(
        "🌐 LANGUAGE:",
        language
    )
    print("=" * 60)
    print()

    return (
        transcript,
        language
    )


# ============================================================
# SARVAM CHAT BRAIN
# ============================================================

def sarvam_chat(
    question: str,
    context: str,
    history: list
):

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is missing."
        )

    messages = []

    messages.append({
        "role": "system",

        "content": """
You are HH Goa AI.

You are a helpful multilingual conversational assistant.

Always answer the user's real question.

You can answer general knowledge, programming,
study questions, technology, science, writing,
casual conversation, jokes and follow-up questions.

Understand English, Telugu, Hindi and other Indian
languages, including mixed and Romanized speech.

Examples:
"java ante enti bro"
"python kya hai"
"machine learning telugu lo cheppu"
"nuvvu em chestav"

Reply naturally in the same language or style
used by the user whenever possible.

Use supplied RAG context only when it is useful.
If the RAG context is unrelated, ignore it completely.

Do not refuse only because the RAG database does
not contain the answer.

If you are uncertain about a fact, say so.
Keep answers clear and easy to understand.
"""
    })

    for item in history[-10:]:

        role = str(
            item.get(
                "role",
                ""
            )
        ).strip()

        content = str(
            item.get(
                "content",
                ""
            )
        ).strip()

        if (
            role in {
                "user",
                "assistant"
            }
            and
            content
        ):

            messages.append({
                "role": role,
                "content": content[:3000]
            })

    if context:

        prompt = f"""
User question:

{question}

Optional RAG context:

{context}

Answer the user's actual question.
Use the context only if it is relevant.
"""

    else:

        prompt = question

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:

        response = requests.post(
            SARVAM_CHAT_URL,

            headers={
                "api-subscription-key":
                    SARVAM_API_KEY,

                "Content-Type":
                    "application/json"
            },

            json={
                "model":
                    SARVAM_CHAT_MODEL,

                "messages":
                    messages,

                "temperature":
                    0.7,

                "top_p":
                    1,

                "max_tokens":
                    600
            },

            timeout=60
        )

    except requests.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect to "
                "Sarvam Chat."
            )
        ) from error

    if not response.ok:

        print()
        print(
            "❌ SARVAM CHAT ERROR:",
            response.status_code
        )

        print(
            response.text[:1200]
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam AI answer failed: "
                f"{response.status_code}"
            )
        )

    data = response.json()

    try:

        answer = (
            data["choices"][0]
            ["message"]["content"]
        ).strip()

    except Exception:

        print(
            "Unexpected Sarvam response:",
            data
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam returned an "
                "unexpected response."
            )
        )

    if not answer:

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam returned an "
                "empty answer."
            )
        )

    return answer


# ============================================================
# VOICE QUERY
# ============================================================

@app.post("/api/voice-query")
async def voice_query(
    file: UploadFile = File(...),
    history: str = Form("[]")
):

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    try:

        history_data = json.loads(
            history
        )

        if not isinstance(
            history_data,
            list
        ):

            history_data = []

    except Exception:

        history_data = []

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    audio_bytes = await file.read()

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "No microphone audio "
                "was received."
            )
        )

    filename = (
        file.filename
        or
        "question.webm"
    )

    mime_type = (
        file.content_type
        or
        "audio/webm"
    )

    mime_type = (
        mime_type
        .split(";")[0]
        .strip()
    )

    # --------------------------------------------------------
    # VOICE -> TEXT
    # --------------------------------------------------------

    question, language = (
        await asyncio.to_thread(
            sarvam_transcribe,
            audio_bytes,
            filename,
            mime_type
        )
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    hits, confidence, retrieval_ms = (
        retrieve(
            question,
            5
        )
    )

    grounded = (
        bool(hits)
        and
        confidence >=
        GROUNDING_THRESHOLD
    )

    context = ""

    if grounded:

        context = "\n\n".join(
            hit["text"][:900]
            for hit in hits[:4]
        )

    # --------------------------------------------------------
    # SARVAM AI
    # --------------------------------------------------------

    answer = await asyncio.to_thread(
        sarvam_chat,
        question,
        context,
        history_data
    )

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    total_ms = (
        time.perf_counter()
        -
        total_start
    ) * 1000

    return {
        "question":
            question,

        "query":
            question,

        "answer":
            answer,

        "detected_language":
            language,

        "grounded":
            grounded,

        "mode":
            (
                "RAG + Sarvam AI"
                if grounded
                else
                "Sarvam AI"
            ),

        "sources": [
            {
                "text":
                    hit["text"][:400],

                "score":
                    round(
                        float(
                            hit["score"]
                        ),
                        4
                    ),

                "source":
                    hit.get(
                        "source",
                        "MSMARCO-XI"
                    )
            }

            for hit in (
                hits[:3]
                if grounded
                else []
            )
        ],

        "timing_ms": {
            "retrieval_ms":
                round(
                    retrieval_ms,
                    2
                ),

            "total_ms":
                round(
                    total_ms,
                    2
                )
        },

        "voice_total_ms":
            round(
                total_ms,
                2
            )
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )