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
# PROJECT PATHS
# Works with:
#
# GitHub / Render:
# main.py
# app.html
# index.html
# chunks.pkl
# faiss.index
#
# AND local structure:
# backend/main.py
# frontend/app.html
# frontend/index.html
# data/chunks.pkl
# data/faiss.index
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent


# Detect frontend location
if (CURRENT_DIR / "index.html").exists():
    FRONTEND_DIR = CURRENT_DIR
else:
    FRONTEND_DIR = PARENT_DIR / "frontend"


# Detect data location
if (CURRENT_DIR / "faiss.index").exists():
    DATA_DIR = CURRENT_DIR
else:
    DATA_DIR = PARENT_DIR / "data"


# Detect project root for .env
if (CURRENT_DIR / ".env").exists():
    ENV_PATH = CURRENT_DIR / ".env"
else:
    ENV_PATH = PARENT_DIR / ".env"


INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

load_dotenv(ENV_PATH, override=True)


# ============================================================
# CONFIGURATION
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
    "https://api.sarvam.ai/speech-to-text"
)

SARVAM_CHAT_URL = (
    "https://api.sarvam.ai/v1/chat/completions"
)


# ============================================================
# APPLICATION STATE
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
    print("=" * 65)
    print("        HH GOA 2026 - VOICE RAG ASSISTANT")
    print("=" * 65)

    print("Frontend directory:", FRONTEND_DIR)
    print("Data directory    :", DATA_DIR)
    print("Index path        :", INDEX_PATH)
    print("Chunks path       :", CHUNKS_PATH)
    print()

    # --------------------------------------------------------
    # Load FAISS and chunks
    # --------------------------------------------------------

    if INDEX_PATH.exists() and CHUNKS_PATH.exists():

        try:

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

            # Warm up embedding model
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

        except Exception as error:

            print(
                "❌ RAG LOAD ERROR:",
                error
            )

    else:

        print("⚠️ RAG files were not found.")

        print(
            "faiss.index exists:",
            INDEX_PATH.exists()
        )

        print(
            "chunks.pkl exists:",
            CHUNKS_PATH.exists()
        )


    # --------------------------------------------------------
    # Sarvam status
    # --------------------------------------------------------

    print()

    if SARVAM_API_KEY:

        print("✅ Sarvam API key configured")

    else:

        print("❌ SARVAM_API_KEY missing")

    print(
        "STT Model :",
        SARVAM_STT_MODEL
    )

    print(
        "Chat Model:",
        SARVAM_CHAT_MODEL
    )

    print("=" * 65)
    print()

    yield


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="HH Goa 2026 Voice RAG",
    description=(
        "Multilingual Voice Enabled "
        "Retrieval Augmented Generation Assistant"
    ),
    version="1.0",
    lifespan=lifespan
)


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def welcome_page():

    index_file = (
        FRONTEND_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        index_file
    )


@app.get("/app")
def assistant_page():

    app_file = (
        FRONTEND_DIR / "app.html"
    )

    if not app_file.exists():

        raise HTTPException(
            status_code=404,
            detail="app.html not found."
        )

    return FileResponse(
        app_file
    )


# Optional script.js support
@app.get("/script.js")
def javascript_file():

    script_file = (
        FRONTEND_DIR / "script.js"
    )

    if not script_file.exists():

        raise HTTPException(
            status_code=404,
            detail="script.js not found."
        )

    return FileResponse(
        script_file,
        media_type="application/javascript"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    vectors = 0

    if state.index is not None:

        vectors = int(
            state.index.ntotal
        )

    return {

        "status":
            "online",

        "vectors":
            vectors,

        "chunks":
            len(state.chunks),

        "rag_ready":
            (
                state.index is not None
                and
                state.model is not None
            ),

        "sarvam_ready":
            bool(
                SARVAM_API_KEY
            ),

        "stt_model":
            SARVAM_STT_MODEL,

        "chat_model":
            SARVAM_CHAT_MODEL
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question: str,
    top_k: int = 5
):

    if (
        state.index is None
        or
        state.model is None
        or
        not state.chunks
    ):

        return (
            [],
            0.0,
            0.0
        )

    start_time = (
        time.perf_counter()
    )

    # Create multilingual query embedding
    embedding = state.model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    top_k = min(
        top_k,
        state.index.ntotal
    )

    scores, ids = (
        state.index.search(
            embedding,
            top_k
        )
    )

    retrieval_ms = (
        (
            time.perf_counter()
            -
            start_time
        )
        *
        1000
    )

    hits = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        idx = int(idx)

        if idx < 0:
            continue

        if idx >= len(
            state.chunks
        ):
            continue

        original_item = (
            state.chunks[idx]
        )

        if isinstance(
            original_item,
            dict
        ):

            item = dict(
                original_item
            )

            text = str(
                item.get(
                    "text",
                    ""
                )
            ).strip()

        else:

            text = str(
                original_item
            ).strip()

            item = {
                "text": text,
                "source": "MSMARCO-XI"
            }

        if not text:
            continue

        item["text"] = text

        item["score"] = float(
            score
        )

        hits.append(
            item
        )

    confidence = 0.0

    if hits:

        confidence = float(
            hits[0]["score"]
        )

    return (
        hits,
        confidence,
        retrieval_ms
    )


# ============================================================
# SARVAM SPEECH TO TEXT
# ============================================================

def sarvam_transcribe(
    audio_bytes: bytes,
    filename: str,
    mime_type: str
):

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "SARVAM_API_KEY "
                "is not configured."
            )
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

        print(
            "SARVAM STT CONNECTION ERROR:",
            error
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect "
                "to Sarvam STT."
            )
        ) from error


    # --------------------------------------------------------
    # Sarvam error
    # --------------------------------------------------------

    if not response.ok:

        print()
        print(
            "❌ SARVAM STT ERROR:",
            response.status_code
        )

        print(
            response.text[:1500]
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam STT failed: "
                f"{response.status_code}"
            )
        )


    # --------------------------------------------------------
    # Sarvam result
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "Invalid response "
                "from Sarvam STT."
            )
        ) from error


    transcript = str(
        data.get(
            "transcript",
            ""
        )
    ).strip()


    language = str(
        data.get(
            "language_code",
            "unknown"
        )
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
    print("=" * 65)

    print(
        "🎤 QUESTION:",
        transcript
    )

    print(
        "🌐 LANGUAGE:",
        language
    )

    print("=" * 65)
    print()


    return (
        transcript,
        language
    )


# ============================================================
# SARVAM CHAT AI
# ============================================================

def sarvam_chat(
    question: str,
    context: str,
    history: list
):

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "SARVAM_API_KEY "
                "is not configured."
            )
        )


    # --------------------------------------------------------
    # System instructions
    # --------------------------------------------------------

    messages = [
        {
            "role":
                "system",

            "content":
                """
You are HH Goa AI, a multilingual conversational
AI assistant created for HH Goa 2026.

Your job is to answer the user's actual question
clearly, naturally and correctly.

You can help with:

- General knowledge
- Programming
- Java
- Python
- Cybersecurity
- Artificial intelligence
- Machine learning
- Computer science
- Mathematics
- Science
- Study questions
- Explanations
- Writing
- Casual conversation
- Follow-up questions

You understand English, Telugu, Hindi and other
Indian languages.

You should also understand mixed speech such as:

"Java ante enti bro?"
"Python lo program rayi"
"AI kya hai?"
"Machine learning telugu lo explain chey"

Reply in the same language or conversational style
as the user whenever practical.

RAG context may be supplied.

IMPORTANT:

Use RAG context only when it is actually relevant
to the user's question.

If the context is unrelated, ignore it and answer
using your general knowledge.

Never make an unrelated RAG passage become the
answer.

If you do not know something reliably, explain
that you are uncertain instead of inventing facts.

For technical and study questions, explain things
in a simple and understandable way.
"""
        }
    ]


    # --------------------------------------------------------
    # Conversation memory
    # --------------------------------------------------------

    if isinstance(
        history,
        list
    ):

        for item in history[-10:]:

            if not isinstance(
                item,
                dict
            ):
                continue

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
                role
                in {
                    "user",
                    "assistant"
                }
                and
                content
            ):

                messages.append({
                    "role":
                        role,

                    "content":
                        content[:4000]
                })


    # --------------------------------------------------------
    # User prompt
    # --------------------------------------------------------

    if context:

        user_prompt = f"""
USER QUESTION:

{question}


OPTIONAL RETRIEVED RAG CONTEXT:

{context}


INSTRUCTIONS:

Answer the user's actual question.

Use the retrieved context only if it helps answer
the question correctly.

If the context is unrelated, ignore it and answer
normally using your own knowledge.
"""

    else:

        user_prompt = question


    messages.append({
        "role":
            "user",

        "content":
            user_prompt
    })


    # --------------------------------------------------------
    # Request Sarvam AI
    # --------------------------------------------------------

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
                    700
            },

            timeout=90
        )

    except requests.RequestException as error:

        print(
            "SARVAM CHAT CONNECTION ERROR:",
            error
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect "
                "to Sarvam AI."
            )
        ) from error


    # --------------------------------------------------------
    # Chat error
    # --------------------------------------------------------

    if not response.ok:

        print()
        print(
            "❌ SARVAM CHAT ERROR:",
            response.status_code
        )

        print(
            response.text[:1500]
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam AI failed: "
                f"{response.status_code}"
            )
        )


    # --------------------------------------------------------
    # Parse answer
    # --------------------------------------------------------

    try:

        data = response.json()

        answer = (
            data["choices"][0]
            ["message"]["content"]
        )

        answer = str(
            answer
        ).strip()

    except Exception as error:

        print()
        print(
            "Unexpected Sarvam response:"
        )

        print(
            response.text[:1500]
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Unexpected response "
                "from Sarvam AI."
            )
        ) from error


    if not answer:

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam AI returned "
                "an empty answer."
            )
        )


    print()
    print(
        "🤖 ANSWER:",
        answer[:500]
    )
    print()


    return answer


# ============================================================
# MAIN VOICE ENDPOINT
# ============================================================

@app.post("/api/voice-query")
async def voice_query(

    file: UploadFile = File(...),

    history: str = Form("[]")
):

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # Parse conversation history
    # ========================================================

    try:

        history_data = (
            json.loads(
                history
            )
        )

        if not isinstance(
            history_data,
            list
        ):

            history_data = []

    except Exception:

        history_data = []


    # ========================================================
    # Read microphone audio
    # ========================================================

    audio_bytes = (
        await file.read()
    )


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


    # Browser may send:
    # audio/webm;codecs=opus
    #
    # Sarvam only needs:
    # audio/webm

    mime_type = (
        mime_type
        .split(";")[0]
        .strip()
    )


    # ========================================================
    # STEP 1: VOICE -> TEXT
    # ========================================================

    stt_start = (
        time.perf_counter()
    )


    question, language = (
        await asyncio.to_thread(
            sarvam_transcribe,
            audio_bytes,
            filename,
            mime_type
        )
    )


    stt_ms = (
        (
            time.perf_counter()
            -
            stt_start
        )
        *
        1000
    )


    # ========================================================
    # STEP 2: RAG RETRIEVAL
    # ========================================================

    hits, confidence, retrieval_ms = (
        retrieve(
            question,
            top_k=5
        )
    )


    grounded = (
        bool(hits)
        and
        confidence
        >=
        GROUNDING_THRESHOLD
    )


    context = ""


    if grounded:

        context_parts = []

        for number, hit in enumerate(
            hits[:4],
            start=1
        ):

            text = hit.get(
                "text",
                ""
            )

            context_parts.append(
                f"[Retrieved Context {number}]\n"
                f"{text[:1200]}"
            )

        context = "\n\n".join(
            context_parts
        )


    # ========================================================
    # STEP 3: SARVAM AI ANSWER
    # ========================================================

    ai_start = (
        time.perf_counter()
    )


    answer = (
        await asyncio.to_thread(
            sarvam_chat,
            question,
            context,
            history_data
        )
    )


    ai_ms = (
        (
            time.perf_counter()
            -
            ai_start
        )
        *
        1000
    )


    # ========================================================
    # TOTAL TIMING
    # ========================================================

    total_ms = (
        (
            time.perf_counter()
            -
            total_start
        )
        *
        1000
    )


    # ========================================================
    # Sources shown only when grounded
    # ========================================================

    sources = []


    if grounded:

        for hit in hits[:3]:

            sources.append({

                "text":
                    hit.get(
                        "text",
                        ""
                    )[:500],

                "score":
                    round(
                        float(
                            hit.get(
                                "score",
                                0
                            )
                        ),
                        4
                    ),

                "source":
                    hit.get(
                        "source",
                        "MSMARCO-XI"
                    )
            })


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "success":
            True,

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

        "confidence":
            round(
                float(
                    confidence
                ),
                4
            ),

        "mode":
            (
                "RAG + Sarvam AI"
                if grounded
                else
                "Sarvam AI"
            ),

        "sources":
            sources,

        "timing_ms": {

            "stt_ms":
                round(
                    stt_ms,
                    2
                ),

            "retrieval_ms":
                round(
                    retrieval_ms,
                    2
                ),

            "ai_ms":
                round(
                    ai_ms,
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
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )