from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import pickle
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# ============================================================
# PATHS
# Works with both:
# Local:  backend/main.py + frontend/ + data/
# Render: main.py + index.html + app.html + chunks.pkl
# ============================================================

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

FRONTEND_DIR = HERE if (HERE / "index.html").exists() else PARENT / "frontend"
DATA_DIR = HERE if (HERE / "chunks.pkl").exists() else PARENT / "data"
ENV_PATH = HERE / ".env" if (HERE / ".env").exists() else PARENT / ".env"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

# Render environment variables take priority.
load_dotenv(ENV_PATH, override=False)

# ============================================================
# CONFIG
# ============================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3").strip()

SARVAM_CHAT_MODEL = os.getenv(
    "SARVAM_CHAT_MODEL",
    "sarvam-105b-conversations"
).strip()

GROUNDING_THRESHOLD = float(
    os.getenv("GROUNDING_THRESHOLD", "0.18")
)

MAX_RAG_CHUNKS = int(
    os.getenv("MAX_RAG_CHUNKS", "6000")
)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"

VECTOR_DIM = 512

# ============================================================
# STATE
# ============================================================

class State:
    chunks: list[Any] = []
    texts: list[str] = []
    index = None


state = State()

# ============================================================
# LIGHTWEIGHT RAG
# ============================================================

def extract_text(item: Any) -> str:

    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item).strip()

    for key in (
        "text",
        "content",
        "chunk",
        "passage",
        "translated_passage",
        "answer",
        "Answer",
        "query",
        "Eng_Query",
        "Eng_Answer",
        "document"
    ):

        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    passages = item.get("passages")

    if isinstance(passages, dict):

        for key in (
            "Translated_passages",
            "English_passages",
            "passage_text"
        ):

            value = passages.get(key)

            if isinstance(value, list):

                text = " ".join(
                    str(x).strip()
                    for x in value
                    if str(x).strip()
                )

                if text:
                    return text

            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def tokenize(text: str) -> list[str]:

    words = re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE
    )

    if not words:
        return []

    features = list(words)

    features.extend(
        f"{words[i]}_{words[i + 1]}"
        for i in range(len(words) - 1)
    )

    return features


def text_to_vector(text: str) -> np.ndarray:

    vector = np.zeros(
        VECTOR_DIM,
        dtype=np.float32
    )

    for token in tokenize(text):

        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=8
        ).digest()

        number = int.from_bytes(
            digest,
            "little",
            signed=False
        )

        slot = number % VECTOR_DIM

        sign = (
            1.0
            if ((number >> 8) & 1) == 0
            else -1.0
        )

        vector[slot] += sign

    norm = float(
        np.linalg.norm(vector)
    )

    if norm > 0:
        vector /= norm

    return vector


def build_rag_index() -> None:

    state.chunks = []
    state.texts = []
    state.index = None

    if not CHUNKS_PATH.exists():

        print(
            f"WARNING: chunks.pkl not found at {CHUNKS_PATH}"
        )

        return

    print(
        f"Loading chunks from {CHUNKS_PATH}"
    )

    with open(CHUNKS_PATH, "rb") as f:
        raw_chunks = pickle.load(f)

    if not isinstance(raw_chunks, list):
        raw_chunks = list(raw_chunks)

    raw_chunks = raw_chunks[:MAX_RAG_CHUNKS]

    clean_chunks: list[Any] = []
    clean_texts: list[str] = []
    vectors: list[np.ndarray] = []

    for item in raw_chunks:

        text = extract_text(item)

        if not text:
            continue

        vec = text_to_vector(text)

        if not np.any(vec):
            continue

        clean_chunks.append(item)
        clean_texts.append(text)
        vectors.append(vec)

    if not vectors:

        print(
            "WARNING: no usable RAG chunks found."
        )

        return

    matrix = np.vstack(
        vectors
    ).astype(np.float32)

    index = faiss.IndexFlatIP(
        VECTOR_DIM
    )

    index.add(matrix)

    state.chunks = clean_chunks
    state.texts = clean_texts
    state.index = index

    print(
        f"RAG READY: {index.ntotal} vectors"
    )


def retrieve(
    question: str,
    top_k: int = 5
):

    if state.index is None or not state.texts:
        return [], 0.0, 0.0

    started = time.perf_counter()

    query_vector = (
        text_to_vector(question)
        .reshape(1, -1)
        .astype(np.float32)
    )

    if not np.any(query_vector):
        return [], 0.0, 0.0

    count = min(
        top_k,
        int(state.index.ntotal)
    )

    scores, ids = state.index.search(
        query_vector,
        count
    )

    retrieval_ms = (
        time.perf_counter() - started
    ) * 1000

    hits = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        idx = int(idx)

        if idx < 0 or idx >= len(state.texts):
            continue

        original = state.chunks[idx]
        text = state.texts[idx]

        source = "MSMARCO-XI"

        if isinstance(original, dict):

            source = str(
                original.get("source")
                or original.get("strategy")
                or original.get("target_lang")
                or "MSMARCO-XI"
            )

        hits.append({
            "text": text,
            "source": source,
            "score": float(score)
        })

    confidence = (
        float(hits[0]["score"])
        if hits
        else 0.0
    )

    return hits, confidence, retrieval_ms

# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("HH GOA 2026 - LIGHTWEIGHT VOICE RAG")
    print("Frontend:", FRONTEND_DIR)
    print("Data:", DATA_DIR)
    print(
        "Sarvam key loaded:",
        "YES" if SARVAM_API_KEY else "NO"
    )
    print("STT model:", SARVAM_STT_MODEL)
    print("Chat model:", SARVAM_CHAT_MODEL)

    try:
        build_rag_index()

    except Exception as error:

        print(
            "RAG STARTUP ERROR:",
            repr(error)
        )

    print("=" * 60)

    yield


app = FastAPI(
    title="HH Goa 2026 Voice RAG",
    version="3.0-light",
    lifespan=lifespan
)

# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    path = FRONTEND_DIR / "index.html"

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(path)


@app.get("/app")
def app_page():

    path = FRONTEND_DIR / "app.html"

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="app.html not found."
        )

    return FileResponse(path)


@app.get("/script.js")
def script_file():

    path = FRONTEND_DIR / "script.js"

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="script.js not found."
        )

    return FileResponse(
        path,
        media_type="application/javascript"
    )


@app.get("/api/health")
def health():

    vectors = (
        int(state.index.ntotal)
        if state.index is not None
        else 0
    )

    return {
        "status": "online",
        "vectors": vectors,
        "chunks": len(state.chunks),
        "rag_ready": state.index is not None,
        "sarvam_ready": bool(SARVAM_API_KEY),
        "mode": "Lightweight FAISS RAG",
        "stt_model": SARVAM_STT_MODEL,
        "chat_model": SARVAM_CHAT_MODEL
    }

# ============================================================
# SARVAM HEADERS
# ============================================================

def sarvam_headers(
    json_request: bool = False
) -> dict[str, str]:

    headers = {
        "api-subscription-key":
            SARVAM_API_KEY
    }

    if json_request:

        headers[
            "Content-Type"
        ] = "application/json"

    return headers

# ============================================================
# SARVAM STT
# ============================================================

def sarvam_transcribe(
    audio_bytes: bytes,
    filename: str,
    mime_type: str
) -> tuple[str, str]:

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is missing."
        )

    try:

        response = requests.post(
            SARVAM_STT_URL,

            headers=sarvam_headers(),

            files={
                "file": (
                    filename,
                    io.BytesIO(audio_bytes),
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
            repr(error)
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect to "
                "Sarvam Speech-to-Text."
            )
        ) from error

    if not response.ok:

        print(
            f"SARVAM STT ERROR "
            f"{response.status_code}: "
            f"{response.text[:1500]}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Sarvam STT failed: "
                f"{response.status_code}"
            )
        )

    try:

        data = response.json()

    except ValueError as error:

        raise HTTPException(
            status_code=502,
            detail=(
                "Sarvam STT returned "
                "invalid JSON."
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
            "language_code"
        )
        or
        "unknown"
    ).strip()

    if not transcript:

        raise HTTPException(
            status_code=422,
            detail=(
                "No speech detected. "
                "Please speak again."
            )
        )

    print(
        "QUESTION:",
        transcript
    )

    print(
        "LANGUAGE:",
        language
    )

    return transcript, language

# ============================================================
# SARVAM CHAT
# ============================================================

def sarvam_chat(
    question: str,
    context: str,
    history: list[Any]
) -> str:

    if not SARVAM_API_KEY:

        raise HTTPException(
            status_code=503,
            detail="SARVAM_API_KEY is missing."
        )

    messages: list[
        dict[str, str]
    ] = [

        {
            "role":
                "system",

            "content":
                (
                    "You are HH Goa AI, a helpful "
                    "multilingual conversational assistant. "

                    "Answer the user's actual question clearly "
                    "and naturally. "

                    "You can answer general knowledge, "
                    "programming, study questions, technology, "
                    "science, mathematics, writing and casual "
                    "conversation. "

                    "Understand English, Telugu, Hindi, mixed "
                    "Indian-language speech and Romanized text. "

                    "Reply in the user's language or style "
                    "when practical. "

                    "Optional RAG context may be provided. "
                    "Use it only when relevant. "

                    "If the context is unrelated, ignore it "
                    "and answer using general knowledge. "

                    "Do not invent facts. "
                    "If uncertain, say so. "

                    "For unsafe requests, do not provide "
                    "harmful instructions."
                )
        }
    ]

    if isinstance(
        history,
        list
    ):

        for item in history[-8:]:

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
                role in {
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
                        content[:3000]
                })

    if context:

        user_message = (
            f"USER QUESTION:\n"
            f"{question}\n\n"

            f"OPTIONAL RETRIEVED "
            f"RAG CONTEXT:\n"
            f"{context}\n\n"

            "Answer the user's actual question. "
            "Use the retrieved context only when "
            "it is relevant."
        )

    else:

        user_message = question

    messages.append({
        "role":
            "user",

        "content":
            user_message
    })

    try:

        response = requests.post(
            SARVAM_CHAT_URL,

            headers=sarvam_headers(
                json_request=True
            ),

            json={
                "model":
                    SARVAM_CHAT_MODEL,

                "messages":
                    messages,

                "temperature":
                    0.6,

                "max_tokens":
                    600
            },

            timeout=90
        )

    except requests.RequestException as error:

        print(
            "SARVAM CHAT CONNECTION ERROR:",
            repr(error)
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not connect "
                "to Sarvam AI."
            )
        ) from error

    if not response.ok:

        print(
            f"SARVAM CHAT ERROR "
            f"{response.status_code}: "
            f"{response.text[:1500]}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"Sarvam AI failed: "
                f"{response.status_code}"
            )
        )

    try:

        data = response.json()

        answer = str(
            data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        ).strip()

    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError
    ) as error:

        print(
            "UNEXPECTED SARVAM RESPONSE:",
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

    print(
        "ANSWER:",
        answer[:500]
    )

    return answer

# ============================================================
# VOICE ENDPOINT
# ============================================================

@app.post(
    "/api/voice-query"
)
async def voice_query(
    file: UploadFile = File(...),
    history: str = Form("[]")
):

    total_start = (
        time.perf_counter()
    )

    try:

        history_data = json.loads(
            history
        )

        if not isinstance(
            history_data,
            list
        ):

            history_data = []

    except (
        json.JSONDecodeError,
        TypeError
    ):

        history_data = []

    audio_bytes = (
        await file.read()
    )

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "No microphone "
                "audio received."
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

    # Speech -> text
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
        time.perf_counter()
        -
        stt_start
    ) * 1000

    # RAG retrieval
    hits, confidence, retrieval_ms = (
        retrieve(
            question,
            top_k=5
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
            (
                f"[Context {n}]\n"
                f"{hit['text'][:1000]}"
            )

            for n, hit in enumerate(
                hits[:4],
                start=1
            )
        )

    # AI answer
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
        time.perf_counter()
        -
        ai_start
    ) * 1000

    total_ms = (
        time.perf_counter()
        -
        total_start
    ) * 1000

    sources = []

    if grounded:

        for hit in hits[:3]:

            sources.append({

                "text":
                    hit[
                        "text"
                    ][:400],

                "score":
                    round(
                        float(
                            hit[
                                "score"
                            ]
                        ),
                        4
                    ),

                "source":
                    hit.get(
                        "source",
                        "MSMARCO-XI"
                    )
            })

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
                "FAISS RAG + Sarvam AI"
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
# LOCAL RUN
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
