from pathlib import Path
import pickle
import re
import time

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MAX_ROWS = 1200
MAX_CHUNKS = 6000


def clean_text(value):

    return re.sub(
        r"\s+",
        " ",
        str(value or "")
    ).strip()


def add_chunk(
    chunks,
    seen,
    text,
    language,
    strategy
):

    text = clean_text(text)

    if len(text) < 25:
        return

    key = text.lower()

    if key in seen:
        return

    seen.add(key)

    chunks.append({
        "text": text,
        "language": language,
        "strategy": strategy,
        "source": "ai4bharat/MSMARCO-XI"
    })


def main():

    started = time.time()

    print()
    print("=" * 60)
    print(" HH GOA - MSMARCO-XI FAST BUILD")
    print("=" * 60)

    print("Loading MSMARCO-XI default stream...")

    dataset = load_dataset(
        "ai4bharat/MSMARCO-XI",
        split="train",
        streaming=True
    )

    chunks = []
    seen = set()

    rows = 0

    for example in dataset:

        rows += 1

        language = clean_text(
            example.get("target_lang")
        ) or "unknown"

        query = clean_text(
            example.get("query")
        )

        answer = clean_text(
            example.get("Answer")
        )

        eng_query = clean_text(
            example.get("Eng_Query")
        )

        eng_answer = clean_text(
            example.get("Eng_Answer")
        )

        passages = (
            example.get("passages")
            or {}
        )

        translated_passages = (
            passages.get(
                "Translated_passages"
            )
            or []
        )

        english_passages = (
            passages.get(
                "English_passages"
            )
            or []
        )

        # --------------------------------
        # Strategy 1: translated Q/A
        # --------------------------------

        if query or answer:

            qa = (
                f"Question: {query}\n"
                f"Answer: {answer}"
            )

            add_chunk(
                chunks,
                seen,
                qa,
                language,
                "metadata_qa"
            )

        # --------------------------------
        # Strategy 2: original English Q/A
        # --------------------------------

        if eng_query or eng_answer:

            english_qa = (
                f"Question: {eng_query}\n"
                f"Answer: {eng_answer}"
            )

            add_chunk(
                chunks,
                seen,
                english_qa,
                "en",
                "english_qa"
            )

        # --------------------------------
        # Strategy 3: translated passages
        # --------------------------------

        for passage in translated_passages[:3]:

            add_chunk(
                chunks,
                seen,
                passage,
                language,
                "translated_passage"
            )

        # --------------------------------
        # Strategy 4: English passages
        # --------------------------------

        for passage in english_passages[:2]:

            add_chunk(
                chunks,
                seen,
                passage,
                "en",
                "english_passage"
            )

        if rows % 100 == 0:

            print(
                f"Rows: {rows} | "
                f"Chunks: {len(chunks)}"
            )

        if (
            rows >= MAX_ROWS
            or
            len(chunks) >= MAX_CHUNKS
        ):
            break

    if not chunks:

        raise RuntimeError(
            "No MSMARCO-XI chunks were created."
        )

    print()
    print(
        "Rows collected:",
        rows
    )

    print(
        "Chunks created:",
        len(chunks)
    )

    print()
    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        item["text"]
        for item in chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print()
    print("=" * 60)
    print("PREPARATION COMPLETE ✅")
    print("Rows    :", rows)
    print("Vectors :", index.ntotal)
    print("Chunks  :", len(chunks))
    print(
        "Time    :",
        round(
            time.time()
            - started,
            2
        ),
        "seconds"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()