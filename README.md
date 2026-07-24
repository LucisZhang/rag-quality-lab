[English](README.md) | [简体中文](README.zh-CN.md)

# 🔬 RAG Quality Lab

A fully local evaluation lab for measuring, comparing, and regression-testing
Retrieval-Augmented Generation pipelines — built to answer the question teams usually skip:
**did that change actually make answer quality better, or did it silently make it worse?**

The lab answered that question concretely once: a routine knowledge-base update — the kind
production RAG systems absorb constantly — **degraded the strongest pipeline on every quality
metric**, and only the regression harness made it visible (§1). That finding is the reason
this repository has since grown from a controlled 12-question testbed into tracked, tested
evaluation infrastructure for an 11,309-document synthetic enterprise corpus (§3).

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A3E85?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-0F766E?style=flat-square)
![Ollama Gemma 4](https://img.shields.io/badge/Ollama%2FGemma%204-Local%20LLM-111827?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)

**Reading paths:** the headline finding and its exact scope are §1–§2; the enterprise-scale
infrastructure is §3; architecture §4; install-and-verify commands §5; evidence rules and data
provenance §6 and [`DATA.md`](DATA.md); extension points §8; repository map §9. This is a
working single-machine evaluation lab, not a released or versioned package, and every result
below carries its own evidence date and scope.

## 1. The finding: a "harmless" KB update degraded quality, and the lab caught it

A knowledge-base update (V1 → V2: two documents revised, one added) **degraded the strongest
pipeline on every quality metric**, most sharply on faithfulness. No code changed. Only the
documents did.

| Metric (Pipeline B, 12-question set) | V1 (baseline) | V2 (updated) | Delta |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.988 | 0.867 | **−0.121** |
| Answer Relevancy | 0.973 | 0.888 | −0.086 |
| Context Precision | 0.913 | 0.850 | −0.063 |
| Context Recall | 0.925 | 0.842 | −0.083 |
| Answer Correctness | 0.921 | 0.846 | −0.075 |

Per-question buckets: **4 degraded / 0 improved / 8 stable** (a question is *degraded* when
any metric drops more than 0.1). Without a regression harness this ships to users unnoticed;
with one, it is a reviewable diff (`results/dashboard_regression/`).

*Evidence:* LLM-judged scores from the **saved 2026-04 run** (local `gemma4:e4b` judge via
Ollama), **deterministically re-parsed 2026-07** by `scripts/verify_a3.py` into
`evidence/verified-2026-07/`. The 4/0/8 buckets and all counts are deterministically
re-verified; the judged scores themselves have not been re-judged — scope and caveats in §6.
These 12-question metrics apply to the controlled set only.

## 2. Why "strongest pipeline" is an evidenced claim: the A/B comparison

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

Pipeline B wins all five metrics (its retrieval recall/hit rate is 1.0 on this set) — treat
the +16.6% lift as a small-set architecture signal, not a general result. Quality is one side
of the decision; the harness measured the cost side too:

| Pipeline | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
| --- | ---: | ---: | ---: | ---: |
| Pipeline A | 28.13 | 37.39 | 39.65 | 29.15 |
| Pipeline B | 128.44 | 188.41 | 212.43 | 134.15 |

Retrieval latency on a cached 50K-document index (100 deterministic queries): Pipeline B pays
roughly 4.6× Pipeline A's latency for its quality lift. Indexing throughput on the same
MS MARCO-derived corpus (`nomic-embed-text` embeddings via Ollama, laptop-class machine):

| Documents | Vectors Indexed | Time (s) | Seconds per 1K Docs |
| ---: | ---: | ---: | ---: |
| 1,000 | 1,121 | 13.22 | 13.22 |
| 5,000 | 5,614 | 60.99 | 12.20 |
| 10,000 | 11,290 | 134.85 | 13.49 |
| 50,000 | 56,039 | 691.17 | 13.82 |

*Evidence:* judged metrics = saved run 2026-04, deterministically re-parsed 2026-07 (means
0.8093 → 0.9438, relative lift +16.6%; artifacts in `results/dashboard_comparison/`).
Latency/indexing timings are deterministic saved artifacts
(`results/scale_performance/*.csv`, 2026-04), re-parsed 2026-07. At ~13.8 s per 1K docs,
embedding the full 498,725-record corpus extrapolates to roughly 1.9 hours on this hardware —
an extrapolation, not a measurement.

## 3. From 12 questions to 11,309 enterprise documents

The 12-question harness proved the point on a corpus small enough to audit by hand: RAG
changes need an evidence-backed quality gate, because harmless-looking changes break things.
The follow-on question was whether the same evaluation lifecycle — versioned datasets,
manifest-verified integrity, deterministic re-verification, model-free CI — holds up on an
enterprise-shaped corpus too large to eyeball. That scale-up is now tracked in this
repository as working infrastructure:

- **Corpus:** **EnterpriseRAG-Bench v1.0.0** (MIT, verified 2026-07-11 on both the GitHub
  LICENSE and the HF dataset card) — a fully synthetic enterprise corpus, free of MS MARCO's
  redistribution constraints. Current scope: confluence + jira slices, **5,189 + 6,120 =
  11,309 documents**, with a **130-question answerable pool** carrying document-level ground
  truth (`expected_doc_ids` at `dsid_*` granularity).
- **Acquisition integrity:** two machines downloaded the six release files independently;
  SHA-256 sums match 6/6 (`evidence/c2-s1-mac-20260712/`,
  `evidence/workstation-c0c1-20260711/`).
- **Deterministic adapters** (`scripts/adapters/`): slices → the lab's KB schema, and
  `questions.jsonl` → the eval schema. The ~88 MB adapted KB stays out of git and regenerates
  **byte-identically** — `data/MANIFEST.json` records the SHA-256 any machine must reproduce.
  Raw-data quirks (duplicate `dsid` ids) are documented, not silently patched
  (`evidence/c2-s1-mac-20260712/`).
- **Judge-free retrieval runner** (`scripts/run_s1_retrieval_ab.py`): document-level ground
  truth makes retrieval precision/recall/hit deterministically scorable with **no LLM judge**.
  Each run records dataset and adapter hashes, exact pipeline definitions, package versions,
  and a manifest hashing every result artifact.
- **Backend seam** (`src/utils.py`): every model access goes through `ollama|hf` factory
  functions selected per component by environment variable, so the same pipelines run on
  machines where Ollama cannot (§5).
- **Tests and CI:** the adapters, runner contract, and backend seam are covered by model-free
  unit tests (`tests/`), and CI (`.github/workflows/ci.yml`) runs lint, tests, and both data
  and artifact verifiers on every push and pull request — no model inference required.

**Scope, stated exactly:** the quality metrics in §1–§2 belong to the saved 12-question runs
only and do **not** transfer to this corpus. At the 11,309-document scale the repository
demonstrates data and evaluation infrastructure — no retrieval or answer-quality results
exist at this scale yet. When they are produced, they will come from the tracked runner with
their own provenance record, in their own labeled tables.

## 4. Architecture

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
| Layer 4. Versioned Knowledge Bases (data/ - see DATA.md)                         |
|   Controlled: V1 (8 docs) / V2 (9 docs) + 12-question eval set (authored)        |
|   Scale: 498,725 MS MARCO-derived passages (out of git) + 500 eval queries       |
|   Enterprise: 11,309 EnterpriseRAG-Bench docs (out of git, adapter-regenerated)  |
|               + 130-question answerable pool (tracked)                           |
+----------------------------------------------------------------------------------+
```

The regression harness (`src/regression_tester.py`) runs one pipeline against both KB versions
and buckets per-question metric deltas into improved / degraded / stable. The EnterpriseRAG
adapters (`scripts/adapters/`) feed Layer 4, so the same pipelines and evaluator run unchanged
against the enterprise corpus.

## 5. Quick start

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

The two large corpora are not in git; regenerate them locally:

```bash
python scale_up_dataset.py                        # streams MS MARCO v2.1 (network required)
python scripts/adapters/enterpriserag_s1_to_kb.py # rebuilds the 11,309-doc enterprise KB;
                                                  # first download+extract the v1.0.0 slices
                                                  # (URLs+checksums: evidence/c2-s1-mac-20260712/)
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
retrieval metrics, regression diffing, the EnterpriseRAG adapters, the retrieval runner's
provenance contract, and backend selection — with scripted fakes in place of the vector
store, cross-encoder, and judge LLM (`tests/dependency_stubs.py` stands in for heavy imports
only when the full environment is absent). GitHub Actions (`.github/workflows/ci.yml`) runs
the same four commands plus the deterministic half of `verify_a3.py` on pushes to `main` and
on pull requests; CI never performs model inference.

## 6. Evidence rules and data provenance

Every results table above carries its label; the rules behind the labels:

- **Deterministic facts** (dataset counts, file checksums, indexing/latency timings,
  saved-file parsing) are re-verified 2026-07 on this copy via `scripts/verify_a3.py` and
  `scripts/verify_data.py`; outputs live in `evidence/verified-2026-07/`.
- **LLM-judged quality metrics** (§1–§2) were produced in the saved 2026-04 runs with a local
  `gemma4:e4b` judge via Ollama and **re-parsed — not re-judged — in 2026-07**. An exact
  same-judge re-run is closed as infeasible on available hardware: the laptop judge is
  impractically slow, and the GPU workstation's NVIDIA driver 470 predates current Ollama's
  CUDA-12 requirement (run record: `evidence/workstation-c0c1-20260711/`). The planned
  replacement is a local judge on the Hugging Face runtime, through the same backend seam as
  §5 — it will test whether the *findings* (B beats A; the V2 update degrades B) replicate
  under an independent judge runtime, reported in its own labeled tables, never blended into
  the 2026-04 columns. Until then, read judged absolute scores from a small local judge as
  relative signals between pipelines/versions, not calibrated absolutes.

Dataset provenance in one line each (full record: [`DATA.md`](DATA.md), integrity values:
`data/MANIFEST.json`):

- Small KBs and the 12-question eval set are **original text authored in `prepare_data.py`**.
- The 191 MB / 498,725-record corpus and the 500-question eval set are **derived from
  Microsoft MS MARCO v2.1** by `scale_up_dataset.py` and stay **out of git**: MS MARCO's
  terms (live-verified 2026-07-11 — "non-commercial research purposes only … without
  extending any license or other intellectual property rights") grant no redistribution
  rights. Clearly-labeled synthetic schema samples stand in (`data/sample_*_synthetic.json`);
  see `DATA.md` §3 for the exact quote and publication rules.
- The EnterpriseRAG-Bench S1 corpus and question pool (§3) are **MIT and fully synthetic**;
  the adapted 130-question set is tracked, the ~88 MB adapted KB regenerates
  deterministically from hash-verified slices.

## 7. Usage guide

**Tab 1 — Interactive Query:** run one question through Pipeline A or B; inspect retrieved
documents (id, title, score) next to the generated answer.

**Tab 2 — Pipeline A/B Comparison:** evaluate both pipelines on the same question set (limit
slider 2–12); outputs a radar chart, grouped bars, per-question score table, and a summary
conclusion. Artifacts land in `results/dashboard_comparison/`.

**Tab 3 — Regression Test:** compare one pipeline across `knowledge_base_v1.json` vs
`knowledge_base_v2.json`. Shows metric-delta table, improved/degraded/stable counts,
per-question status cards (V1 vs V2 answers, retrieved doc ids, per-metric deltas), and a
KB-change summary (added/modified/removed/unchanged docs). Thresholds: degraded if any metric
< −0.1, improved if any metric > +0.1, else stable. Artifacts land in
`results/dashboard_regression/`.

**Tab 4 — Scale & Performance:** indexing benchmark over 1K/5K/10K/50K corpus subsets and a
retrieval-only latency benchmark (100 deterministic queries against the cached 50K index).
Artifacts land in `results/scale_performance/`.

For the enterprise corpus, the judge-free retrieval A/B runs from the command line:

```bash
python scripts/run_s1_retrieval_ab.py --help   # pipelines, question caps, output dir
```

## 8. Extending with custom pipelines

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

## 9. Project structure

```text
rag-quality-lab/
├── app.py                        # Streamlit dashboard (four modes)
├── src/
│   ├── utils.py                  # model backend seam (ollama|hf), JSON loading, chunking
│   ├── rag_pipelines.py          # BaseRAGPipeline, NaiveVectorRAG, HybridRerankRAG
│   ├── evaluation_engine.py      # RAGEvaluator (RAGAS + local-judge fallback)
│   └── regression_tester.py      # RegressionTester and report generation
├── data/
│   ├── knowledge_base_v1.json    # 8-doc baseline KB (authored)
│   ├── knowledge_base_v2.json    # 9-doc updated KB (authored)
│   ├── eval_questions.json       # 12-question controlled eval set (authored)
│   ├── eval_questions_regression_debug.json  # 4-question debug subset
│   ├── eval_questions_enterpriserag_s1.json  # 130-question enterprise pool (MIT, adapted)
│   ├── sample_*_synthetic.json   # labeled synthetic schema samples
│   └── MANIFEST.json             # checksums / sizes / record counts
├── results/                      # saved 2026-04 experiment artifacts (CSV/JSON)
├── evidence/                     # verification outputs + acquisition records
├── scripts/
│   ├── verify_a3.py              # deterministic re-checks of saved artifacts
│   ├── verify_data.py            # data integrity vs MANIFEST.json
│   ├── run_s1_retrieval_ab.py    # judge-free enterprise retrieval A/B (deterministic)
│   ├── adapters/                 # EnterpriseRAG-Bench -> lab schema converters
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

(The full corpora, Chroma stores, venvs, and generated runtime artifacts are gitignored by
design — see `.gitignore` and `docs/A1_COPY_NOTES.md`.)

## 10. Technology stack

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
`nomic-embed-text`, Ollama endpoint `http://127.0.0.1:11434`; the `hf` backend (§5) swaps
these for env-named Hugging Face models. Pipeline B additionally downloads
`cross-encoder/ms-marco-MiniLM-L-6-v2` on first run.

## 11. Limitations (honest scope)

1. **Judge fidelity and freshness.** Quality metrics come from a small local judge
   (`gemma4:e4b` via Ollama) in saved 2026-04 runs, re-parsed but not re-judged in 2026-07;
   an exact same-judge re-run is closed as infeasible on available hardware (§6). Until the
   planned HF-runtime judge lane runs, treat absolute scores cautiously — deltas between
   pipelines and KB versions are the meaningful signal.
2. **Controlled evaluation size.** The high-fidelity A/B and regression workflow runs on a
   small hand-authored 12-question set; the 500-question set drives scale benchmarks, not
   judged quality runs.
3. **No quality results at enterprise scale yet.** The 11,309-document scope (§3) is
   evidence-verified data and evaluation infrastructure; its first retrieval numbers will
   come from `scripts/run_s1_retrieval_ab.py` with their own provenance record.
4. **Single-machine scope.** Everything runs on one local machine; the measured scale story
   tops out at a 50K-document index on a laptop.
5. **License-gated data publication.** MS MARCO's verified terms (non-commercial research
   only, no redistribution rights — `DATA.md` §3) mean MS MARCO-derived content stays out of
   any public release; the repo documents schemas via synthetic samples instead.
6. **No UI screenshots yet.** Dashboard screenshots will be captured from a live session and
   added; none are reconstructed or mocked in the meantime.

## 12. References

- *Corrective Retrieval Augmented Generation*. arXiv, 2024. <https://arxiv.org/abs/2401.15884>
- *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv, 2023. <https://arxiv.org/abs/2309.15217>
- *Okapi at TREC-3*. NIST TREC-3 Proceedings, 1994. <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>
- *Passage Re-ranking with BERT*. arXiv, 2019. <https://arxiv.org/abs/1901.04085>
- *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv, 2020. <https://arxiv.org/abs/2002.10957>
- *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv, 2016. <https://arxiv.org/abs/1611.09268>

## 13. Rights

No open-source license is currently granted for this repository; all rights reserved. Dataset
licenses are governed separately and strictly: see `DATA.md` for the MS MARCO non-commercial,
no-redistribution terms this repository is bound by, and the MIT license of the
EnterpriseRAG-Bench slices used by the tracked enterprise-scale data and evaluation
infrastructure.
