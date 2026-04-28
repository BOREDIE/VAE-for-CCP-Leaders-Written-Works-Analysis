import os
import re
import csv

import pandas as pd


def clean_chinese(raw: str) -> str:
    raw = re.sub(r"^.*?={60}\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"感谢.*$",       "", raw, flags=re.DOTALL)
    raw = re.sub(r"[^一-鿿　-〿＀-￯\n]", " ", raw)
    return raw.strip()


def chunk_text(text: str, size: int, stride: int, min_len: int) -> list:
    return [text[s:s + size] for s in range(0, len(text), stride)
            if len(text[s:s + size]) >= min_len]


def load_corpus(hp: dict) -> pd.DataFrame:
    records = []
    with open(hp["meta_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    demo_map = {}
    if hp.get("demo_dir") and os.path.isdir(hp["demo_dir"]):
        for fname in os.listdir(hp["demo_dir"]):
            core = re.sub(r"^\d+_", "", fname).replace("_htm.txt", ".htm.txt")
            demo_map[core] = os.path.join(hp["demo_dir"], fname)

    for row in rows:
        filename = row["filename"]
        basename = os.path.basename(filename)
        raw = None

        if hp.get("data_dir"):
            full = os.path.join(hp["data_dir"], filename)
            if os.path.exists(full):
                raw = open(full, encoding="utf-8").read()

        if raw is None and basename in demo_map:
            raw = open(demo_map[basename], encoding="utf-8").read()

        if raw is None:
            continue

        text = clean_chinese(raw)
        if len(text) >= hp["min_chunk"]:
            records.append({
                "filename": filename,
                "author":   row["author_english"],
                "date":     row["date"],
                "text":     text,
            })

    return pd.DataFrame(records)


def build_chunks(df_docs: pd.DataFrame, hp: dict) -> pd.DataFrame:
    chunk_records = []
    for _, doc in df_docs.iterrows():
        for i, raw_chunk in enumerate(
            chunk_text(doc["text"], hp["chunk_size"], hp["chunk_stride"], hp["min_chunk"])
        ):
            if len(raw_chunk) >= hp["min_chunk"]:
                chunk_records.append({
                    "doc_id":   doc["filename"],
                    "author":   doc["author"],
                    "date":     doc["date"],
                    "chunk_id": i,
                    "text":     raw_chunk,
                })
    return pd.DataFrame(chunk_records)
