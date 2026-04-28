import os
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pipeline selector ──────────────────────────────────────────────────────
# "tfidf"  — TF-IDF char-bigrams, BCE loss, CPU-friendly
# "qwen3"  — Qwen3-Embedding-4B dense embeddings, Cosine loss, GPU recommended
# "custom" — Any HuggingFace sentence-transformer; set CUSTOM_* variables below
PIPELINE = "tfidf"

CUSTOM_MODEL_ID = "BAAI/bge-m3"
CUSTOM_EMBED_DIM = 1024
CUSTOM_EMBED_INSTRUCTION = (
    "Given a short passage from a 1930s Chinese Communist Party document, "
    "retrieve passages with similar ideological stance and rhetorical style"
)

AUTHOR_META = {
    "Wang Ming":    {"color": "#BA7517", "marker": "D", "faction": "28 Bolsheviks"},
    "Bo Gu":        {"color": "#F06292", "marker": "X", "faction": "28 Bolsheviks"},
    "Zhang Guotao": {"color": "#7F77DD", "marker": "P", "faction": "Disaligned"},
    "Mao Zedong":   {"color": "#E24B4A", "marker": "o", "faction": "Maoist"},
    "Zhou Enlai":   {"color": "#378ADD", "marker": "s", "faction": "Maoist"},
    "Liu Shaoqi":   {"color": "#1D9E75", "marker": "^", "faction": "Maoist"},
}

HP = {
    # ── Paths ─────────────────────────────────────────────────────────────
    "meta_csv": "/path/to/metadata.csv",        # ← update before running
    "data_dir": "/path/to/marxists_downloads",  # ← update before running
    "demo_dir": "/path/to/marxists_downloads",  # ← update before running
    "out_dir":  "/path/to/output",              # ← update before running

    # ── Chunking ──────────────────────────────────────────────────────────
    "chunk_size":   300,
    "chunk_stride": 150,
    "min_chunk":    100,

    # ── TF-IDF (used when PIPELINE == "tfidf") ────────────────────────────
    "tfidf_max_features": 5000,
    "tfidf_ngram":        (1, 2),
    "tfidf_min_df":       2,
    "tfidf_max_df":       0.97,
    "tfidf_sublinear_tf": True,

    # ── Embedding (used when PIPELINE in {"qwen3", "custom"}) ─────────────
    "qwen_model_id":     "Qwen/Qwen3-Embedding-4B",
    "embed_dim":         512,
    "embed_batch_size":  4,
    "embed_max_length":  128,
    "embed_device":      "cuda",
    "embed_instruction": (
        "Given a short passage from a 1930s Chinese Communist Party document, "
        "retrieve passages with similar ideological stance and rhetorical style"
    ),
    "embed_cache_file":  "/tmp/embed_cache_{model}.npy",
    "embed_label_cache": "/tmp/embed_label_{model}.npy",

    # ── Jieba (embedding pipelines only) ──────────────────────────────────
    "jieba_user_dict": None,

    # ── VAE architecture ──────────────────────────────────────────────────
    "hidden_dim": 512,
    "depth":      3,
    "dropout":    0.1,
    "latent_dim": 16,   # overridden in main.py based on PIPELINE
    "beta":       0.5,  # overridden in main.py based on PIPELINE
    "free_bits":  0.0,  # overridden in main.py based on PIPELINE

    # ── Training ──────────────────────────────────────────────────────────
    "learning_rate":    1e-3,
    "kl_warmup_epochs": 20,
    "epochs":           80,
    "batch_size":       64,
    "lr_gamma":         0.97,
    "weight_decay":     1e-5,
    "grad_clip":        5.0,

    # ── Visualisation ─────────────────────────────────────────────────────
    "tsne_perplexity": 30,
    "tsne_iter":       1000,
}
