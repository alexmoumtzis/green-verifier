# GreenVerifier

GreenVerifier is a retrieval-augmented, encoder-based framework for automated verification of ESG claims in long-form corporate sustainability and TCFD reports.  
Given a claim and its source report, the system classifies the claim as **Supported** or **Not Supported**, while identifying the underlying textual or numerical evidence.

The pipeline consists of multi-source document ingestion, dense retrieval of candidate evidence via a Qdrant vector store, and supervised verification using a DeBERTa cross-encoder fine-tuned with Multiple Instance Learning (MIL).

---

## Final Report

The project report is available here:

- [GreenVerifier Final Report (PDF)](./GreenVerifier_final_report.pdf)



## Project Structure

```
green-verifier/
├── ingestion/          # Multi-source document collection
│   ├── config.py       # Centralised env-var configuration
│   ├── db.py           # PostgreSQL helpers (company metadata)
│   ├── s3io.py         # S3 upload / hashing utilities
│   ├── edgar.py        # SEC EDGAR 10-K & 20-F downloader
│   ├── gri.py          # GRI sustainability report scraper
│   ├── company_pdf.py  # Website fallback ESG PDF crawler
│   └── ingest.py       # Orchestrator (EDGAR → GRI → fallback)
│
├── labeling/           # Claim generation via RAG
│   ├── rag_pipline.py      # Local LLM RAG (LangChain + FAISS)
│   └── rag_pipeline_aws.py # Cloud LLM claim generation (HF Inference + S3)
│
├── model/              # Core ML pipeline
│   ├── stage0.py           # Document vectorisation & Qdrant indexing
│   ├── stage1.py           # Dense retrieval of evidence candidates
│   ├── split_dataset.py    # Company-level train/dev/test split (70/15/15)
│   ├── stage2_training.py  # DeBERTa MIL fine-tuning
│   ├── stage2_inference.py # Base-model inference & metrics
│   └── stage2_evaluate.py  # Fine-tuned model evaluation
│
├── evaluation/         # Baselines & metrics
│   ├── baseline_llama.py       # LLaMA / Qwen zero-/few-shot baseline
│   ├── baseline_openai.py      # GPT few-shot + CoT baseline
│   ├── evaluation_baseline.py  # Metric computation (sklearn)
│   └── get_test_data.py        # Download splits from S3
│
├── results/            # Generated artefacts (JSON)
├── run_stage*.slurm    # SLURM job scripts
├── requirements.txt
└── README.md
```

---

## Key Technologies

| Component | Technology | Details |
|---|---|---|
| **Embedding model** | `intfloat/e5-base-v2` | 768-dim dense vectors |
| **Cross-encoder** | `microsoft/deberta-v3-base` | 2-class, MIL fine-tuned |
| **Claim generation** | Meta-Llama-3-8B-Instruct | Via HF Inference API |
| **Vector store** | Qdrant Cloud | Cosine distance; `reports_text` + `reports_kpi` collections |
| **Object storage** | AWS S3 | Documents, models, results |
| **Database** | PostgreSQL | Company metadata |
| **Baselines** | Qwen3-4B / GPT | Zero-shot, few-shot, chain-of-thought |

---

## Installation

```bash
conda create -n greenverifier python=3.10
conda activate greenverifier
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file (or export the variables) with the following:

```bash
# PostgreSQL
PG_HOST=...
PG_DB=...
PG_USER=...
PG_PASSWORD=...
PG_PORT=5432

# AWS S3
AWS_REGION=...
S3_BUCKET_RAW=...
S3_PREFIX_EDGAR=edgar
S3_PREFIX_SITE=site
S3_OUTPUT_PREFIX=results

# SEC EDGAR
SEC_USER_AGENT="YourName your@email.com"

# Qdrant
QDRANT_URL=...
QDRANT_API_KEY=...

# LLM / Inference
HF_TOKEN=...          # Hugging Face access token
HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
API_KEY=...            # OpenAI key (baselines only)

# Optional
ENABLE_TEXTRACT_OCR=0  # Set to 1 for AWS Textract OCR on scanned PDFs
```

---

## Running the Pipeline

GreenVerifier is organised as a three-stage pipeline. Each stage can be executed independently but must be run in order.

### On a SLURM cluster

```bash
sbatch run_stage0.slurm           # Vectorisation & indexing
sbatch run_stage1.slurm           # Dense retrieval
sbatch run_stage2_training.slurm  # MIL fine-tuning
sbatch run_stage2_inference.slurm # Base-model inference
sbatch run_stage2_evaluate.slurm  # Fine-tuned model evaluation
```

### Locally

```bash
python model/stage0.py            # Vectorisation & indexing
python model/stage1.py            # Dense retrieval
python model/split_dataset.py     # Train / dev / test split
python model/stage2_training.py   # MIL fine-tuning
python model/stage2_inference.py  # Base-model inference
python model/stage2_evaluate.py   # Fine-tuned model evaluation
```

---

## Pipeline Details

### Stage 0 — Document Indexing & Claim Generation

1. **Ingestion** (`ingestion/ingest.py`) downloads reports from SEC EDGAR (10-K, 20-F), GRI sustainability profiles, and company websites as a fallback.
2. **Vectorisation** (`model/stage0.py`) extracts text and structured KPIs (iXBRL facts, tables, numeric lines), chunks text at 400 tokens with 100-token overlap, embeds with E5, and upserts into two Qdrant collections:
   - `reports_text` — general text chunks  
   - `reports_kpi` — structured financial/KPI facts
3. **Claim generation** (`labeling/rag_pipeline_aws.py`) prompts a LLaMA-3 model to produce 6–10 claims per report with balanced support/not-support labels.

### Stage 1 — Dense Retrieval

`model/stage1.py` embeds each claim with E5, queries Qdrant filtered by the source report, and retrieves the **top-5** candidate evidence passages from both collections. Output: `batch_results_retrieval.json`.

### Stage 2 — Verification

- **Training** (`model/stage2_training.py`): Fine-tunes DeBERTa v3-base using MIL with logsumexp soft-pooling. Each training bag contains a claim paired with its retrieved candidates plus 2 random negative chunks. Only the top 4 transformer layers and the classifier head are unfrozen.
- **Inference** (`model/stage2_inference.py`): Runs the base (pre-trained) cross-encoder with argmax aggregation: $\hat{y} = \arg\max_y \max_i P(y \mid \text{claim}, e_i)$.
- **Evaluation** (`model/stage2_evaluate.py`): Same aggregation using the MIL-fine-tuned model. Reports accuracy, precision, recall, and F1.

### Baselines

| Baseline | Model | Strategies |
|---|---|---|
| `evaluation/baseline_llama.py` | Qwen3-4B-Instruct (4-bit) | Zero-shot, few-shot, CoT |
| `evaluation/baseline_openai.py` | GPT | Few-shot + CoT |

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Bag size | 4 |
| Gradient accumulation steps | 8 (effective batch 32) |
| Learning rate | 1 × 10⁻⁵ |
| Weight decay | 0.01 |
| Warmup | 10 % of total steps |
| Max sequence length | 512 tokens |
| Unfrozen layers | Top 4 + classifier |
| Mixed precision | bfloat16 (AMP) |
| Negative samples per claim | 2 |

---

## SLURM Resources

| Job | GPUs | CPUs | RAM | Time |
|---|---|---|---|---|
| Stage 0 (indexing) | 2 | 8 | 32 GB | 6 h |
| Stage 1 (retrieval) | 2 | 8 | 32 GB | 6 h |
| Stage 2 (training) | 1 × V100 | 16 | 64 GB | 12 h |
| Stage 2 (inference) | 2 | 8 | 32 GB | 6 h |
| Stage 2 (evaluation) | 2 | 8 | 32 GB | 6 h |

All jobs load GCC 13.3.0, Python 3.11.9, and CUDA 12.4.0.

---

## License

This project is provided for academic and research purposes.
