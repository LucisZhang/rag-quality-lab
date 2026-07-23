[English](README.md) | [简体中文](README.zh-CN.md)

# 🔬 RAG Quality Lab

A fully local evaluation platform for measuring, comparing, and regression-testing
Retrieval-Augmented Generation pipelines — built to answer the question teams usually skip:
**did that change actually make answer quality better, or did it silently make it worse?**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A3E85?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-0F766E?style=flat-square)
![Ollama Gemma 4](https://img.shields.io/badge/Ollama%2FGemma%204-Local%20LLM-111827?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)

## 0. Current state at a glance (2026-07)

The repository now contains two deliberately separated evidence layers:

- **Historical quality baseline:** the controlled 12-question A/B and regression workflow,
  saved 2026-04 judged runs, and deterministic 2026-07 re-verification under
  [`evidence/verified-2026-07/`](evidence/verified-2026-07/README.md). These small-set metrics
  remain the only published answer-quality results.
- **EnterpriseRAG-Bench C2 scale-up:** tracked adapters, manifest contracts, 130 answerable
  questions with document-level ground truth, backend-aware retrieval seams, and model-free
  tests for processing **11,309 synthetic enterprise documents** — 5,189 Confluence records
  plus 6,120 Jira records. The roughly 88 MB adapted knowledge base stays out of Git and is
  regenerated deterministically from hash-verified, MIT-licensed EnterpriseRAG-Bench v1.0.0
  slices.

C2 proves the data and evaluation infrastructure, not retrieval or answer quality. The later
C3 A/B timebox ended with an explicit no-result record rather than a metric from a substitute
pipeline; no retrieval, answer-quality, judged, or fallback C3 result should be inferred. The
historical 12-question comparison in §3 does **not** transfer to the 11,309-document scope.
That is the operating rule of this lab: no metric is better than a metric produced by the wrong
stack.

**Where to start, by reader:** inspect the 11,309-document processing path in §6 and
[`DATA.md`](DATA.md); evaluate historical claims in §2 and
[`evidence/verified-2026-07/`](evidence/verified-2026-07/README.md); run the lab in §7; or extend
the implementation through §5, §9, `src/`, `scripts/adapters/`, and `tests/`. This is a working
single-machine evaluation lab, not a released or versioned package, and every result carries its
own evidence date and scope.

## 1. The headline finding: a "harmless" KB update degraded quality, and the lab caught it

A knowledge-base update (V1 → V2: two documents revised, one added — the kind of routine
content refresh production RAG systems absorb constantly) **degraded the strongest pipeline
on every quality metric**, most sharply on faithfulness. No code changed. Only the documents did.

| Metric (Pipeline B, 12-question set) | V1 (baseline) | V2 (updated) | Delta |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.988 | 0.867 | **−0.121** |
| Answer Relevancy | 0.973 | 0.888 | −0.086 |
| Context Precision | 0.913 | 0.850 | −0.063 |
| Context Recall | 0.925 | 0.842 | −0.083 |
| Answer Correctness | 0.921 | 0.846 | −0.075 |

Per-question buckets: **4 degraded / 0 improved / 8 stable** (a question is *degraded* when any
metric drops more than 0.1). Without a regression harness this ships to users unnoticed; with
one, it is a reviewable diff (`results/dashboard_regression/`).

*Evidence label:* LLM-judged scores from the **saved run 2026-04** (local `gemma4:e4b` judge),
**deterministically re-parsed 2026-07** (`scripts/verify_a3.py` →
`evidence/verified-2026-07/`); **fresh judged re-verification pending (workstation)**. The
4/0/8 buckets and all counts are deterministically re-verified.

## 2. Evidence status — read this before quoting any number

This repo distinguishes two kinds of claims, and every results table below carries its label:

- **Deterministic facts** (dataset counts, file checksums, indexing/latency timings, saved-file
  parsing): re-verified 2026-07 on this copy via `scripts/verify_a3.py` and
  `scripts/verify_data.py`; outputs in `evidence/verified-2026-07/`.
- **LLM-judged quality metrics** (faithfulness, relevancy, precision, recall, correctness):
  produced in the **saved 2026-04 runs** with a local `gemma4:e4b` judge via Ollama, and
  **re-parsed — not re-judged — in 2026-07**. A 2026-07-11 workstation C0/C1 run proved the
  public clone, CI, the EnterpriseRAG-Bench S1 data path, and a PyTorch/HF CUDA embedding
  smoke — but Ollama itself could not run there (the host's NVIDIA driver 470 predates
  current Ollama's CUDA-12 requirement), so an **exact same-judge re-run of the 2026-04
  baseline is closed as infeasible** on available hardware (2026-07-12 decision; the
  laptop-judge path was closed earlier for speed). The fresh judged lane is re-scoped to a
  **Hugging Face-runtime local judge (Lane L2)** via this repo's `ollama|hf` backend seam:
  it tests whether the *findings* (B beats A; the V2 update degrades B) replicate under an
  independent judge runtime, and its scores will be reported in their own labeled tables,
  never blended into the 2026-04 columns. Run records live under `evidence/`
  (`evidence/workstation-c0c1-20260711/` where present; the S1 acquisition record ships in
  `evidence/c2-s1-mac-20260712/`). Judged absolute scores from a small local judge should
  be read as relative signals between pipelines/versions, not calibrated absolutes.

Dataset provenance, licensing status, and integrity checksums live in [`DATA.md`](DATA.md)
and `data/MANIFEST.json`.

## 3. Pipeline A/B comparison — quality vs. latency, both sides measured

Same 12-question evaluation set, same knowledge base, same judge; only the retrieval
architecture differs:

- **Pipeline A — NaiveVectorRAG:** Chroma similarity search → top-k contexts → Gemma 4.
- **Pipeline B — HybridRerankRAG:** Chroma top-10 + BM25 top-10 → merge/deduplicate →
  cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → Gemma 4.

| Metric | Pipeline A (Naive Vector) | Pipeline B (Hybrid+Rerank) | Delta |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.888 | 0.988 | +0.100 |
| Answer Relevancy | 0.804 | 0.973 | +0.169 |
| Context Precision | 0.784 | 0.913 | +0.129 |
| Context Recall | 0.800 | 0.925 | +0.125 |
| Answer Correctness | 0.771 | 0.921 | +0.150 |
| **Overall Mean** | **0.809** | **0.944** | **+0.135 (+16.6% relative)** |

*All five metrics in this table come from the 12-question controlled set only — treat the
+16.6% lift as a small-set architecture signal, not a general result.*

Pipeline B wins all five metrics; its retrieval-diagnostic recall/hit rate is 1.0 on this set.
The price appears below in §4: retrieval latency roughly 4.6× Pipeline A's at the 50K-document
scale. That pair of numbers — quality lift and its latency cost — is the architecture decision
this lab exists to make measurable.

*Evidence label:* judged metrics = saved run 2026-04, deterministically re-parsed 2026-07
(mean 0.8093 → 0.9438, relative lift +16.6%); fresh judged re-verification pending
(workstation). Source artifacts: `results/dashboard_comparison/`.

## 4. Scale benchmarks (deterministic timings, saved run 2026-04)

Indexing throughput on the MS MARCO-derived corpus (embeddings: `nomic-embed-text` via Ollama,
local machine):

| Documents | Vectors Indexed | Time (s) | Seconds per 1K Docs |
| ---: | ---: | ---: | ---: |
| 1,000 | 1,121 | 13.22 | 13.22 |
| 5,000 | 5,614 | 60.99 | 12.20 |
| 10,000 | 11,290 | 134.85 | 13.49 |
| 50,000 | 56,039 | 691.17 | 13.82 |

Retrieval latency on the cached 50K-document index (100 deterministic queries):

| Pipeline | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| Pipeline A | 28.13 | 37.39 | 39.65 | 29.15 |
| Pipeline B | 128.44 | 188.41 | 212.43 | 134.15 |

*Evidence label:* timings are deterministic saved artifacts
(`results/scale_performance/*.csv`, 2026-04), re-parsed 2026-07. They characterize a laptop-class
machine; at ~13.8 s per 1K docs, embedding the full 498,725-record corpus extrapolates to
roughly 1.9 hours — the quantified argument for GPU-workstation runs (extrapolation, not a
measurement).

## 5. Architecture

```text
+----------------------------------------------------------------------------------+
| Layer 1. Streamlit Dashboard (app.py)                                            |
|   Interactive Query | Pipeline A/B Comparison | Regression Test | Scale & Perf   |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| Layer 2. Evaluation Engine (src/evaluation_engine.py)                            |
|   RAGEvaluator - five RAGAS quality metrics (faithfulness, answer relevancy,     |
|   context precision, context recall, answer correctness)                         |
|   + retrieval diagnostics (retrieval_precision / recall / hit)                   |
|   Backend modes: auto | ragas | fallback (local-LLM judge)                       |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| Layer 3. RAG Pipelines (src/rag_pipelines.py)                                    |
|   A: NaiveVectorRAG   - Chroma similarity -> top-k -> Gemma 4                    |
|   B: HybridRerankRAG  - Chroma top-10 + BM25 top-10 -> merge/dedupe              |
|                         -> cross-encoder rerank -> Gemma 4                       |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| Layer 4. Versioned Knowledge Base (data/ - see DATA.md)                          |
|   Controlled: V1 (8 docs) / V2 (9 docs) + 12-question eval set (authored)        |
|   Scale: 498,725 MS MARCO-derived passages (out of git) + 500 eval queries       |
+----------------------------------------------------------------------------------+
```

The regression harness (`src/regression_tester.py`) runs one pipeline against both KB versions
and buckets per-question metric deltas into improved / degraded / stable.

## 6. Data & licensing (summary — the full record is `DATA.md`)

- Small KBs and the 12-question eval set are **original text authored in `prepare_data.py`**.
- The 191 MB / 498,725-record corpus and the 500-question eval set are **derived from Microsoft
  MS MARCO v2.1** by `scale_up_dataset.py`. Both stay **out of git** in the public repo;
  integrity travels via `data/MANIFEST.json` expected counts + `scripts/verify_data.py`.
- **MS MARCO terms live-verified 2026-07-11**: "non-commercial research purposes only …
  without extending any license or other intellectual property rights" (`microsoft/msmarco`
  `Notice.md`, read via GitHub API). No corpus records are republished here; clearly-labeled
  synthetic schema samples stand in (`data/sample_*_synthetic.json`). See `DATA.md` §3 for
  the exact quote and the publication rules this repo is bound by.

### EnterpriseRAG-Bench S1 (Track C scale-up — data + adapters ready, no results yet)

The scale-up track moves evaluation onto **EnterpriseRAG-Bench v1.0.0** (MIT, verified on
both the GitHub LICENSE and HF card 2026-07-11) — a fully synthetic enterprise corpus, so it
is free of MS MARCO's redistribution constraints. Current scope is **S1 = confluence + jira**:
11,309 documents and a **130-question answerable pool** with document-level ground truth
(`expected_doc_ids` at `dsid_*` granularity), which enables **judge-free deterministic
retrieval evaluation** (`scripts/run_s1_retrieval_ab.py`) ahead of any judged run.

- Acquisition integrity: two machines downloaded the six release files independently and
  their SHA-256 sums match exactly (`evidence/c2-s1-mac-20260712/`).
- Adapters (`scripts/adapters/`) convert slices → the lab's KB schema and `questions.jsonl`
  → the eval schema deterministically; `data/MANIFEST.json` carries both outputs' integrity
  (the ~88 MB adapted KB stays out of git and regenerates byte-identically).
- **Status honestly:** no S1 retrieval or quality results exist yet — the first numbers come
  from the planned GPU-workstation run (C3). Nothing here claims otherwise.

## 7. Quick start

### Prerequisites

- Python 3.11 (pinned in `.python-version`)
- [Ollama](https://ollama.com/)
- Enough free disk for local models, Chroma persistence, and benchmark artifacts

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt   # exact pinned versions
# (requirements.txt is the smaller top-level spec; the lockfile is the reproducible path)
```

See `docs/A2_ENVIRONMENT.md` for how the lockfile was produced and the smoke-test record.

### Pull local models

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama serve   # if not already running
```

### Model backends (`ollama` default, `hf` optional)

With no configuration, everything uses the local Ollama models above — unchanged behavior.
On machines where Ollama cannot run (e.g., a GPU host whose NVIDIA driver predates current
Ollama's CUDA-12 requirement), select the Hugging Face backend per component:

```bash
export RAG_MODEL_BACKEND=hf              # both components, or use per-component:
export RAG_EMBEDDING_BACKEND=hf          #   embeddings only (retrieval-only runs)
export RAG_HF_LLM_MODEL=<org/model-id>       # required for the hf LLM
export RAG_HF_EMBEDDING_MODEL=<org/model-id> # required for hf embeddings
# optional: RAG_HF_DEVICE=cuda:0  RAG_HF_MAX_NEW_TOKENS=512  RAG_HF_NORMALIZE_EMBEDDINGS=1
```

There are deliberately **no default HF model ids**: verify the model card (license,
size/VRAM) before the first download, then export the id. The `hf` backend needs the full
lockfile environment (`langchain-huggingface` is pinned there); CI never exercises it.

### Data

Small controlled datasets ship in `data/`. Verify integrity any time:

```bash
python scripts/verify_data.py
```

Expected result: the script re-checks every present data file against `data/MANIFEST.json`
and exits non-zero on any mismatch (`DATA.md` §4), so a clean exit means the tracked data
matches the manifest.

The large corpus is not in git; regenerate it (network required, streams MS MARCO v2.1):

```bash
python scale_up_dataset.py
```

### Launch

```bash
streamlit run app.py
```

Expected result: Streamlit starts and serves the dashboard shell on a local port. The
recorded 2026-07-10 headless smoke test (`docs/A2_ENVIRONMENT.md`) bound `127.0.0.1:8521`
and answered `HTTP/1.1 200 OK`.

### Re-verify the saved evidence

```bash
python scripts/verify_a3.py   # deterministic re-checks -> evidence/verified-2026-07/
```

### Run the unit tests (no models required)

```bash
pip install -r requirements-ci.txt   # lightweight: no torch/chromadb/ollama
pytest                               # model-free logic; all LLM calls mocked
ruff check .                         # lint
python scripts/verify_data.py        # data integrity vs data/MANIFEST.json
```

The tests cover the model-free core — chunking, hybrid BM25/dense merge-dedupe-rerank,
retrieval metrics, and regression diffing — with scripted fakes in place of the vector store,
cross-encoder, and judge LLM (`tests/dependency_stubs.py` stands in for heavy imports only
when the full environment is absent). GitHub Actions (`.github/workflows/ci.yml`) runs the
same four commands plus the deterministic half of `verify_a3.py` on pushes to `main` and on
pull requests; CI never performs model inference.

## 8. Usage guide

**Tab 1 — Interactive Query:** run one question through Pipeline A or B; inspect retrieved
documents (id, title, score) next to the generated answer.

**Tab 2 — Pipeline A/B Comparison:** evaluate both pipelines on the same question set (limit
slider 2–12); outputs a radar chart, grouped bars, per-question score table, and a summary
conclusion. Artifacts land in `results/dashboard_comparison/`.

**Tab 3 — Regression Test:** compare one pipeline across `knowledge_base_v1.json` vs
`knowledge_base_v2.json`. Shows metric-delta table, improved/degraded/stable counts, per-question
status cards (V1 vs V2 answers, retrieved doc ids, per-metric deltas), and a KB-change summary
(added/modified/removed/unchanged docs). Thresholds: degraded if any metric < −0.1, improved if
any metric > +0.1, else stable. Artifacts land in `results/dashboard_regression/`.

**Tab 4 — Scale & Performance:** indexing benchmark over 1K/5K/10K/50K corpus subsets and a
retrieval-only latency benchmark (100 deterministic queries against the cached 50K index).
Artifacts land in `results/scale_performance/`.

## 9. Extending with custom pipelines

Pipelines share one contract (`src/rag_pipelines.py`):

```python
def query(self, query: str, k: int = 4) -> dict[str, Any]:
    return {
        "question": query,
        "answer": answer,
        "contexts": [doc.page_content for doc in retrieved_docs],
        "retrieved_doc_ids": retrieved_doc_ids,
    }
```

Subclass `BaseRAGPipeline` and implement:

```python
def retrieve(self, query: str, k: int = 4) -> list[Document]: ...
```

To add a Pipeline C: implement the class, add a cached factory in `app.py`, register it in
`PIPELINE_CLASS_MAP` and `PIPELINE_APP_CONFIG`, and update the pipeline selectors. (The A/B
comparison flow is currently hard-coded to two pipelines; generalizing
`compute_comparison_evaluation()` and its charts is the known extension point.)

`RAGEvaluator.load_eval_dataset()` requires each eval item to have `question`, `ground_truth`,
and `relevant_doc_ids` (which may be empty, as in the regenerated
`data/large_eval_questions.json`). Data file schemas are documented with examples in `DATA.md`
and the synthetic samples.

## 10. Project structure

```text
rag-quality-lab/
├── app.py                        # Streamlit dashboard (four modes)
├── src/
│   ├── utils.py                  # Ollama clients, JSON loading, chunking, Chroma creation
│   ├── rag_pipelines.py          # BaseRAGPipeline, NaiveVectorRAG, HybridRerankRAG
│   ├── evaluation_engine.py      # RAGEvaluator (RAGAS + local-judge fallback)
│   └── regression_tester.py      # RegressionTester and report generation
├── data/
│   ├── knowledge_base_v1.json    # 8-doc baseline KB (authored)
│   ├── knowledge_base_v2.json    # 9-doc updated KB (authored)
│   ├── eval_questions.json       # 12-question controlled eval set (authored)
│   ├── eval_questions_regression_debug.json  # 4-question debug subset
│   ├── eval_questions_enterpriserag_s1.json  # 130-question S1 pool (MIT, adapted)
│   ├── sample_*_synthetic.json   # labeled synthetic schema samples
│   └── MANIFEST.json             # checksums / sizes / record counts
├── results/                      # saved 2026-04 experiment artifacts (CSV/JSON)
├── evidence/                     # verification outputs + acquisition records
├── scripts/
│   ├── verify_a3.py              # deterministic re-checks of saved artifacts
│   ├── verify_data.py            # data integrity vs MANIFEST.json
│   ├── run_s1_retrieval_ab.py    # judge-free S1 retrieval A/B (deterministic)
│   ├── adapters/                 # EnterpriseRAG-Bench S1 -> lab schema converters
│   └── ci/run_verify_a3_deterministic.py  # CI wrapper (stubs heavy deps)
├── tests/                        # model-free unit tests (LLM calls mocked)
├── tools/                        # dashboard asset-export helper scripts
├── .github/workflows/ci.yml      # lint + tests + verifiers; no model inference
├── docs/                         # A1 copy notes, A2 environment notes
├── DATA.md                       # dataset provenance, licensing, verification protocol
├── prepare_data.py               # builds (and IS the source text of) the small datasets
├── scale_up_dataset.py           # streams MS MARCO v2.1 -> large corpus + eval set
├── pyproject.toml                # ruff + pytest configuration
├── requirements.txt              # top-level dependency spec
├── requirements-ci.txt           # lightweight CI/test dependencies
└── requirements-lock-py311.txt   # pinned reproducibility lockfile
```

(The full corpus, Chroma stores, venvs, and generated runtime artifacts are gitignored by
design — see `.gitignore` and `docs/A1_COPY_NOTES.md`.)

## 11. Technology stack

| Component | Technology | Version (lockfile) | Purpose |
| --- | --- | --- | --- |
| Language | Python | 3.11 | Core implementation language |
| Dashboard | Streamlit | 1.45.1 | Interactive local UI |
| RAG framework | LangChain | 0.3.25 | Prompting, documents, orchestration primitives |
| LangChain integrations | `langchain-community` | 0.3.24 | Community integrations |
| Ollama integration | `langchain-ollama` | 0.3.3 | Local LLM and embedding access |
| Vector store integration | `langchain-chroma` | 0.2.6 | LangChain wrapper for Chroma |
| Vector database | ChromaDB | 1.5.7 | Persistent vector retrieval |
| Evaluation framework | RAGAS | 0.2.15 | Automated RAG evaluation |
| Dataset loader | `datasets` | 3.6.0 | Streaming MS MARCO ingestion |
| Lexical retrieval | `rank-bm25` | 0.2.2 | BM25 candidate retrieval |
| Reranker | `sentence-transformers` | 4.1.0 | Cross-encoder reranking |
| Visualization | Plotly | 6.1.2 | Charts and benchmark plots |
| Data analysis | Pandas | 2.2.3 | Tabular result processing |
| Numerical utilities | NumPy | 1.26.4 | Numeric support |
| Also pinned | `langchain-huggingface`, `openpyxl` | 0.2.0, 3.1.5 | HF embeddings wrapper, Excel export |

Default model configuration (`src/utils.py`, `ollama` backend): LLM `gemma4:e4b`, embeddings
`nomic-embed-text`, Ollama endpoint `http://127.0.0.1:11434`; the `hf` backend (§7) swaps
these for env-named Hugging Face models. Pipeline B additionally downloads
`cross-encoder/ms-marco-MiniLM-L-6-v2` on first run.

## 12. Limitations (honest scope)

1. **Judge fidelity and freshness.** Quality metrics come from a small local judge
   (`gemma4:e4b` via Ollama) in saved 2026-04 runs. They were deterministically re-parsed in
   2026-07 but **not yet re-judged** — and an exact same-judge re-run is now **closed as
   infeasible**: the laptop is impractically slow (1-question probe, 2026-07) and the GPU
   workstation's NVIDIA driver 470 cannot run current Ollama (2026-07-11 run; path retired
   2026-07-12). The replacement is a fresh **HF-runtime judge lane (Lane L2)** that tests
   whether the findings replicate under an independent judge runtime; until it runs, treat
   absolute scores cautiously — deltas between pipelines and KB versions are the meaningful
   signal.
2. **Controlled evaluation size.** The high-fidelity A/B and regression workflow runs on a
   small hand-authored 12-question set; the 500-question set drives scale benchmarks, not
   judged quality runs.
3. **Single-machine scope.** Everything runs on one local machine; the scale story tops out at
   a 50K-document index on a laptop. Larger-scale runs are a planned workstation track with
   its own evidence discipline.
4. **License-gated data publication.** MS MARCO's verified terms (non-commercial research
   only, no redistribution rights — `DATA.md` §3) mean MS MARCO-derived content stays out of
   any public release; the repo documents schemas via synthetic samples instead.
5. **No UI screenshots yet.** Dashboard screenshots will be captured from a live session and
   added; none are reconstructed or mocked in the meantime.

## 13. References

- *Corrective Retrieval Augmented Generation*. arXiv, 2024. <https://arxiv.org/abs/2401.15884>
- *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv, 2023. <https://arxiv.org/abs/2309.15217>
- *Okapi at TREC-3*. NIST TREC-3 Proceedings, 1994. <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>
- *Passage Re-ranking with BERT*. arXiv, 2019. <https://arxiv.org/abs/1901.04085>
- *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv, 2020. <https://arxiv.org/abs/2002.10957>
- *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv, 2016. <https://arxiv.org/abs/1611.09268>

## 14. Rights

No open-source license is currently granted for this repository; all rights reserved. Dataset
licenses are governed separately and strictly: see `DATA.md` for the MS MARCO non-commercial,
no-redistribution terms this repository is bound by, and the MIT license of the
EnterpriseRAG-Bench slices used by the tracked C2 data and evaluation infrastructure.
