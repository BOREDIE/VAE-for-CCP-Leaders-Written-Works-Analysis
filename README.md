# CCP Ideal-Point Estimation via β-VAE

**Exploratory Ideal-Point Estimation of Chinese Communist Party Leadership Using Variational Auto-Encoder**

*Boxuan Yu*

> **Note (2026-08-28):** This β-VAE approach has been superseded. See [CCP-Ideal-Point-Estimation](https://github.com/BOREDIE/CCP-Ideal-Point-Estimation) for the current Wordfish/Wordscores-based analysis and the reasoning behind the change. This repository is left unchanged as the exact replication record for the original course paper.

---

## Overview

This project applies an unsupervised β-Variational Autoencoder (β-VAE) to a corpus of 286 writings authored by six CCP Politburo members between 1930 and 1944, a period of intense intra-party factional rivalry between the Maoist and Comintern-aligned (28 Bolsheviks) factions. The goal is to test whether latent ideological divisions documented by historians are recoverable from the texts alone, without any factional labels during training.

**Research question:** *Do 1930s–1940s CCP leadership writings reflect the historically documented factional divisions that represent unidentical patterns of ideological interpretation?*

The six authors span three factions:

| Faction | Authors |
|---|---|
| Maoist | Mao Zedong, Zhou Enlai, Liu Shaoqi |
| Comintern (28 Bolsheviks) | Wang Ming, Bo Gu |
| Disaligned | Zhang Guotao |

Two feature-extraction pipelines are compared:

- **TF-IDF** — character bigrams weighted by TF-IDF, 5 000-dimensional sparse vectors, BCE reconstruction loss. Captures surface lexical style.
- **Qwen3-Embedding-4B / custom** — dense sentence embeddings produced by a HuggingFace model with jieba segmentation, cosine reconstruction loss. Captures semantic meaning.

Results show that the TF-IDF pipeline yields stronger factional separation (silhouette score and t-SNE clustering), suggesting factional identity is encoded in lexical surface style more than in semantic content.

---

## Repository Layout

```
ccp_vae/
├── config.py        ← the only file you need to edit
├── data.py          ← corpus loading and chunking
├── features.py      ← TF-IDF / Qwen3 / custom embedding extraction
├── model.py         ← Encoder, Decoder, BetaVAE
├── train.py         ← training loop and sanity check
├── evaluate.py      ← silhouette score, PCA, t-SNE, hyperparameter sweep
├── visualize.py     ← all plot functions (saves PNG files)
├── checkpoint.py    ← save/load model checkpoints, latent CSV, hyperparams JSON
├── main.py          ← CLI entry point that wires everything together
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

For the embedding pipelines only (not needed for `tfidf`):

```bash
pip install transformers>=4.51.0 sentencepiece jieba
```

A CUDA-capable GPU is strongly recommended for `qwen3` and `custom` pipelines. The `tfidf` pipeline runs comfortably on CPU.

---

## Data Preparation

### Required files

1. **Raw text files** — plain-text `.txt` exports of the source documents, one file per article. The scraping scripts in the parent directory (`scrape_marxists.py`, `extract_bogu.py`, etc.) can download and extract these automatically.

2. **Metadata CSV** — a single CSV file that maps each text file to its author and date. It must have at least these three columns:

   | Column | Description |
   |---|---|
   | `filename` | path to the text file, relative to `data_dir` |
   | `author_english` | author name exactly as it appears in `AUTHOR_META` in `config.py` |
   | `date` | publication date (any string format is accepted) |

   Example row:
   ```
   filename,author_english,date
   mao/on_practice.txt,Mao Zedong,1937-07
   ```

### Text file format

Each text file should be a raw UTF-8 export. The loader automatically:
- strips everything before a line of 60 `=` characters (standard Marxists.org header)
- removes acknowledgement lines beginning with 感谢
- strips non-Chinese characters

No manual cleaning is required.

---

## Configuration — edit `config.py` only

**All user-facing settings live in `config.py`. You should not need to touch any other file for a standard run.**

### 1. Set your paths

```python
HP = {
    "meta_csv": "/path/to/metadata.csv",        # ← your metadata CSV
    "data_dir": "/path/to/marxists_downloads",  # ← folder containing text files
    "demo_dir": "/path/to/marxists_downloads",  # ← same as data_dir, or a fallback folder
    "out_dir":  "/path/to/output",              # ← where plots, checkpoints, and CSVs are saved
    ...
}
```

### 2. Choose your pipeline

```python
PIPELINE = "tfidf"   # "tfidf" | "qwen3" | "custom"
```

| Value | Input | Loss | Recommended hardware |
|---|---|---|---|
| `"tfidf"` | 5 000-d char-bigram TF-IDF | BCE | CPU |
| `"qwen3"` | 512-d Qwen3-Embedding-4B | Cosine | GPU |
| `"custom"` | dim you specify | Cosine | GPU |

For `"custom"`, also set:

```python
CUSTOM_MODEL_ID        = "BAAI/bge-m3"   # any HuggingFace sentence-transformer
CUSTOM_EMBED_DIM       = 1024
CUSTOM_EMBED_INSTRUCTION = "..."         # task instruction prepended to each chunk
```

### 3. Adjust hyperparameters (optional)

Sensible defaults are already filled in. The ones most worth changing are:

```python
HP["latent_dim"]        # size of the latent space z (default: 16 for tfidf, 32 for embedding)
HP["beta"]              # KL regularisation weight  (default: 0.5 for tfidf, 0.25 for embedding)
HP["epochs"]            # training epochs           (default: 80)
HP["chunk_size"]        # characters per chunk      (default: 300)
HP["tfidf_max_features"]# vocabulary size           (default: 5000)
```

### 4. Add or change authors

Edit `AUTHOR_META` to add new authors or change faction labels, colours, and plot markers:

```python
AUTHOR_META = {
    "Author Name": {"color": "#hex", "marker": "o", "faction": "Faction Label"},
    ...
}
```

The metadata CSV's `author_english` column must match the keys here exactly.

---

## Running

### Standard run

```bash
python main.py
```

This reads all settings from `config.py` and runs the full pipeline: load → chunk → extract features → train → evaluate → visualise → save.

### Override settings from the command line

Any value set in `config.py` can also be overridden without editing the file:

```bash
python main.py \
  --pipeline  tfidf \
  --meta-csv  /data/metadata.csv \
  --data-dir  /data/texts \
  --out-dir   /outputs \
  --epochs    80 \
  --latent-dim 16 \
  --beta      0.5
```

### Run with hyperparameter sweep

Add `--sweep` to run a grid search over latent dimensions and β values before the main training run. The best configuration is printed at the end:

```bash
python main.py --sweep
```

---

## Outputs

All files are written to `HP["out_dir"]`:

| File | Description |
|---|---|
| `{pipeline}_vae_training_curves.png` | Total ELBO, reconstruction loss, KL divergence, KL warmup schedule |
| `{pipeline}_vae_latent_projections.png` | PCA and t-SNE 2-D projections of the latent space, coloured by author/faction |
| `{pipeline}_vae_dim_activations.png` | Per-dimension activation histograms, one row per latent dimension |
| `{pipeline}_vae_sweep.png` | Silhouette, reconstruction, and KL vs z and β (only with `--sweep`) |
| `{pipeline}_vae_checkpoint.pt` | PyTorch model checkpoint (weights + hyperparams + training history) |
| `{pipeline}_vae_latent_coords.csv` | Per-chunk latent coordinates (z₀…z_k, PCA x/y, t-SNE x/y, author, faction) |
| `{pipeline}_vae_hyperparams.json` | Full hyperparameter snapshot |

---

## Model Architecture

The β-VAE encodes each text chunk into a low-dimensional latent space *z* and reconstructs the input from *z*:

- **Encoder**: 3 fully-connected layers (width 512, ReLU, dropout 0.1) → μ(x) and log σ²(x)
- **Latent sampling**: z = μ + ε · exp(½ log σ²), ε ~ N(0, I)
- **Decoder**: symmetric 3-layer network → reconstructed input
- **Loss**: β-weighted ELBO = reconstruction loss + β · KL(q(z|x) ‖ N(0,I))
  - BCE reconstruction for TF-IDF (inputs in [0, 1])
  - Cosine reconstruction for embeddings (L2-normalised vectors)
- **KL warmup**: weight increases linearly from 0 to β over the first 20 epochs
- **Free-bits**: minimum KL cost of 0.05 nats per dimension (embedding pipelines only)

---

## Key references

- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with A Constrained Variational Framework. *ICLR*.
- Kingma & Welling (2013). Auto-Encoding Variational Bayes. *ICLR*.
- Imai, Lo & Olmsted (2016). Fast Estimation of Ideal Points with Massive Data. *APSR*, 110(4).
- Hua, Mosher & Jian (2018). Seizing the Power of Ideological "Interpretation." In *How the Red Sun Rose*. CUHK Press.
