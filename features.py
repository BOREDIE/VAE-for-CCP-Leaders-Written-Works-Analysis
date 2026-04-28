import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset


# ── TF-IDF ────────────────────────────────────────────────────────────────

def extract_tfidf(df_chunks, hp):
    print("[TF-IDF] Extracting char-bigram features ...", flush=True)
    vec = TfidfVectorizer(
        analyzer      = "char_wb",
        ngram_range   = hp["tfidf_ngram"],
        max_features  = hp["tfidf_max_features"],
        min_df        = hp["tfidf_min_df"],
        max_df        = hp["tfidf_max_df"],
        sublinear_tf  = hp["tfidf_sublinear_tf"],
        norm          = "l2",
        strip_accents = None,
        decode_error  = "ignore",
    )
    X_sparse = vec.fit_transform(df_chunks["text"].tolist())
    X_np = X_sparse.toarray().astype(np.float32)
    print(f"  Feature matrix : {X_np.shape}", flush=True)
    print(f"  Sparsity       : {(X_np == 0).mean():.1%} zeros", flush=True)

    feature_names = vec.get_feature_names_out()
    print("  Top 10 bigrams per author:", flush=True)
    for a in sorted(df_chunks["author"].unique()):
        mask  = df_chunks["author"].values == a
        means = X_np[mask].mean(axis=0)
        top   = means.argsort()[::-1][:10]
        print(f"    {a:<15}: {' '.join(feature_names[top])}", flush=True)

    return X_np, vec


# ── Shared embedding helpers ───────────────────────────────────────────────

def _jieba_segment(text: str) -> str:
    import jieba
    tokens = jieba.cut(text, cut_all=False)
    return " ".join(t for t in tokens if t.strip() and not re.match(r"^[\s　-〿]+$", t))


def _embed_with_last_token(model_id: str, seg_texts: list, hp: dict, device: torch.device) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel

    embed_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"  Loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    mdl = AutoModel.from_pretrained(model_id, torch_dtype=embed_dtype).to(device)
    mdl.eval()

    instruction = hp["embed_instruction"]
    bs          = hp["embed_batch_size"]
    max_len     = hp["embed_max_length"]
    target_dim  = hp["embed_dim"]
    prefixed    = [f"Instruct: {instruction}\nQuery: {t}" for t in seg_texts]
    all_emb     = []
    n = len(prefixed)
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, n, bs):
            batch = prefixed[i:i + bs]
            bd = tok(batch, padding=True, truncation=True, max_length=max_len,
                     return_tensors="pt", padding_side="left")
            bd = {k: v.to(device) for k, v in bd.items()}
            out = mdl(**bd)
            am  = bd["attention_mask"]
            lhs = out.last_hidden_state
            # last-token pooling (handles both left- and right-padding)
            if (am[:, -1].sum() == am.shape[0]):
                emb = lhs[:, -1]
            else:
                sl  = am.sum(dim=1) - 1
                emb = lhs[torch.arange(lhs.shape[0], device=lhs.device), sl]
            emb = F.normalize(emb, p=2, dim=1)
            if target_dim < emb.shape[1]:
                emb = F.normalize(emb[:, :target_dim], p=2, dim=1)
            all_emb.append(emb.cpu().float().numpy())
            if (i // bs + 1) % 50 == 0:
                print(f"    {i + len(batch)}/{n} ({time.time() - t0:.0f}s)", flush=True)

    return np.vstack(all_emb)


def _embed_with_mean_pool(model_id: str, seg_texts: list, hp: dict, device: torch.device,
                          custom_embed_dim: int, custom_embed_instruction: str) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel

    embed_dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id, torch_dtype=embed_dtype).to(device)
    mdl.eval()

    prefixed = [f"Instruct: {custom_embed_instruction}\nQuery: {t}" for t in seg_texts]
    all_emb  = []
    bs = hp["embed_batch_size"]
    n  = len(prefixed)

    with torch.no_grad():
        for i in range(0, n, bs):
            batch = prefixed[i:i + bs]
            bd = tok(batch, padding=True, truncation=True,
                     max_length=hp["embed_max_length"], return_tensors="pt")
            bd   = {k: v.to(device) for k, v in bd.items()}
            out  = mdl(**bd)
            mask = bd["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb  = F.normalize(emb[:, :custom_embed_dim], p=2, dim=1)
            all_emb.append(emb.cpu().float().numpy())

    return np.vstack(all_emb)


def _load_or_compute_cache(cache_emb, cache_lbl, n_expected, compute_fn):
    if os.path.exists(cache_emb) and os.path.exists(cache_lbl):
        print("  Cache found — loading ...", flush=True)
        X_np = np.load(cache_emb)
        if len(X_np) == n_expected:
            return X_np
        print("  Stale cache — re-embedding ...", flush=True)
        os.remove(cache_emb)
        os.remove(cache_lbl)

    X_np = compute_fn()
    return X_np


# ── Qwen3-Embedding-4B pipeline ───────────────────────────────────────────

def extract_qwen3(df_chunks, hp):
    import jieba
    jieba.setLogLevel("ERROR")
    if hp.get("jieba_user_dict") and os.path.exists(hp["jieba_user_dict"]):
        jieba.load_userdict(hp["jieba_user_dict"])

    MODEL_ID   = hp["qwen_model_id"]
    _model_tag = MODEL_ID.split("/")[-1]
    cache_emb  = hp["embed_cache_file"].format(model=_model_tag)
    cache_lbl  = hp["embed_label_cache"].format(model=_model_tag)

    df_chunks = df_chunks.copy()
    df_chunks["seg_text"] = df_chunks["text"].apply(_jieba_segment)
    seg_texts = df_chunks["seg_text"].tolist()

    embed_device = torch.device(hp["embed_device"] if torch.cuda.is_available() else "cpu")

    def _compute():
        X = _embed_with_last_token(MODEL_ID, seg_texts, hp, embed_device)
        np.save(cache_emb, X)
        np.save(cache_lbl, df_chunks["author"].values)
        print(f"  Saved cache → {cache_emb}", flush=True)
        return X

    X_np = _load_or_compute_cache(cache_emb, cache_lbl, len(df_chunks), _compute)
    hp["embed_dim"] = X_np.shape[1]
    print(f"  Embedding matrix : {X_np.shape}", flush=True)
    return X_np, None


# ── Custom HuggingFace embedding pipeline ─────────────────────────────────

def extract_custom(df_chunks, hp, custom_model_id, custom_embed_dim, custom_embed_instruction):
    import jieba
    jieba.setLogLevel("ERROR")

    MODEL_ID   = custom_model_id
    _model_tag = MODEL_ID.replace("/", "_")
    cache_emb  = hp["embed_cache_file"].format(model=_model_tag)
    cache_lbl  = hp["embed_label_cache"].format(model=_model_tag)

    hp["qwen_model_id"]     = MODEL_ID
    hp["embed_dim"]         = custom_embed_dim
    hp["embed_instruction"] = custom_embed_instruction

    df_chunks = df_chunks.copy()
    df_chunks["seg_text"] = df_chunks["text"].apply(_jieba_segment)
    seg_texts = df_chunks["seg_text"].tolist()

    embed_device = torch.device(hp["embed_device"] if torch.cuda.is_available() else "cpu")
    print(f"[Custom] Using {MODEL_ID}  dim={custom_embed_dim}", flush=True)

    def _compute():
        X = _embed_with_mean_pool(MODEL_ID, seg_texts, hp, embed_device,
                                   custom_embed_dim, custom_embed_instruction)
        np.save(cache_emb, X)
        np.save(cache_lbl, df_chunks["author"].values)
        print(f"  Saved → {cache_emb}", flush=True)
        return X

    X_np = _load_or_compute_cache(cache_emb, cache_lbl, len(df_chunks), _compute)
    hp["embed_dim"] = X_np.shape[1]
    print(f"  Embedding matrix : {X_np.shape}", flush=True)
    return X_np, None


# ── Shared: tensor + DataLoader construction ──────────────────────────────

def build_tensors(X_np, df_chunks, hp, device):
    authors  = sorted(df_chunks["author"].unique())
    a2i      = {a: i for i, a in enumerate(authors)}
    y_np     = np.array([a2i[a] for a in df_chunks["author"]], dtype=np.int64)
    X_tensor = torch.from_numpy(X_np).to(device)
    y_tensor = torch.from_numpy(y_np).to(device)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=True, drop_last=False)
    hp["input_dim"] = X_np.shape[1]
    return X_tensor, y_tensor, y_np, authors, a2i, loader
