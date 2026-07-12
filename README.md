# RAG Quality Lab

RAG Quality Lab is a local evaluation workbench for comparing retrieval pipelines,
reviewing knowledge-base changes, and building evidence contracts around RAG experiments.

## Current public status

Read this section before quoting a number or treating this repository as the latest
evidence checkpoint.

- The current public repository baseline is commit `0fc1433`.
- A later local evidence checkpoint, `6c887a1`, prepared the C2 EnterpriseRAG-Bench S1
  adapters, deterministic manifests, backend-aware verifier, runner contracts, and tests.
- Those C2 commits are not yet synced to this public repository.
- The verified C2 evidence floor is 11,309 S1 documents, 130 S1-answerable questions,
  and 68 passing tests.
- C2 verifies dataset and evaluation infrastructure. It does not establish retrieval or
  answer quality.
- The C3 retrieval A/B timebox ended before execution. It produced no metrics and no
  fallback or toy-pipeline result.

The portfolio claim registry is the current authority until the C2 code sync is reviewed
and completed. Do not describe this public branch as containing C2 yet.

## Current claim boundary

| Claim | Status | What it proves |
| --- | --- | --- |
| 11,309 EnterpriseRAG-Bench S1 documents | Verified at local checkpoint `6c887a1` | Deterministic corpus adaptation and manifest integrity |
| 130 S1-answerable questions | Verified at local checkpoint `6c887a1` | Deterministic eval-set adaptation and manifest integrity |
| 68 passing tests | Verified locally on 2026-07-12 | Model-free logic, adapters, manifests, and runner contracts |
| C3 retrieval results | Not generated | No retrieval metric table exists |
| Current answer-quality result | Not claimed | No fresh judged run supports one |
| C2 code in this public branch | Pending | Public HEAD remains `0fc1433` |

Historical saved artifacts may remain in the baseline project for audit and code-path
inspection. Their old metric values are not the current portfolio claim floor and should
not be promoted as fresh or C2/C3 results.

## What this public baseline contains

The baseline application demonstrates the project shape:

- a Streamlit interface for interactive query inspection, pipeline comparison,
  regression review, and scale/performance workflows;
- `NaiveVectorRAG` and `HybridRerankRAG` pipeline implementations;
- a versioned small knowledge base and authored evaluation questions;
- deterministic verification scripts for tracked data and saved artifacts;
- synthetic schema samples that avoid republishing restricted corpus records.

This source demonstrates implementation capability. It does not, by itself, verify a
current model-quality claim.

## Architecture

```text
Streamlit review surface
        |
        v
Evaluation and regression contracts
        |
        +--> Pipeline A: dense retrieval
        |
        +--> Pipeline B: dense + BM25 + reranking
        |
        v
Versioned knowledge bases and evidence manifests
```

The pipeline interface returns the question, answer, retrieved contexts, and document
identifiers. Evaluation and regression code consumes that common contract so a retrieval
change or knowledge-base update can be reviewed consistently.

## C2 evidence checkpoint

The unsynced local C2 work uses EnterpriseRAG-Bench v1.0.0 S1, limited to its synthetic
Confluence and Jira sources:

- Confluence: 5,189 documents
- Jira: 6,120 documents
- Combined S1 scope: 11,309 documents
- S1-answerable pool: 130 questions with document-level ground truth

The C2 work adds deterministic adapters, manifest hashes, a backend seam, and a
judge-free retrieval-runner contract. The adapted knowledge base stays out of Git because
of size and regenerates from hash-verified source slices. The 130-question adapted eval
set is MIT-licensed and can be tracked when the C2 sync is approved.

These are data and infrastructure claims only.

## C3 no-results record

The intended C3 comparison was the real dense pipeline versus the real hybrid/rerank
pipeline over the full S1 scope. Preflight found that the local environment lacked the
required Chroma, PyTorch, Transformers, sentence-transformers, LangChain integrations,
and cached cross-encoder.

Installing the heavy stack exceeded the approved offline and dependency-size timebox.
Replacing the real pipelines with an easier lexical substitute would have invalidated the
comparison, so the run stopped without metrics. No answer-quality, retrieval, judged, or
fallback result should be inferred.

## Baseline quick start

Prerequisites:

- Python 3.11
- Ollama for the baseline local model path
- sufficient local disk for models and generated indexes

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt
python scripts/verify_data.py
pytest
ruff check .
streamlit run app.py
```

Model downloads, corpus generation, judged evaluation, and large indexing are explicit
local actions. They are not CI steps and should not be run merely to reproduce this README.

## Data and publication boundaries

- The small controlled knowledge bases and questions are authored in the project.
- MS MARCO-derived corpus and query records stay out of the public repository. Synthetic
  examples document their schema without redistributing source content.
- EnterpriseRAG-Bench S1 is synthetic and MIT-licensed, but the adapted knowledge base
  remains out of Git for size and repository hygiene.
- Dataset counts and hashes belong in `data/MANIFEST.json`; descriptive claims belong in
  `DATA.md`; current public claims must agree with both.

## Public sync checklist

Before describing the public repository as C2-complete:

1. Review the prepared C2 diff against public commit `0fc1433`.
2. Confirm that no raw corpus, model cache, private path, key, or generated result is added.
3. Run Ruff, Pytest, `scripts/verify_data.py`, and the deterministic verifier in a clean clone.
4. Reconcile this README, `DATA.md`, and `data/MANIFEST.json` in the same commit.
5. Push only after the separate publication approval.

## Limitations

- Public HEAD is baseline code; local C2 work is not yet public.
- No current retrieval or answer-quality metric is claimed.
- C3 produced no result artifact.
- Heavy model execution and large-corpus indexing are environment-dependent.
- Archived judged scores are historical evidence, not fresh verification.

## License and sources

See `DATA.md` for dataset-specific provenance and terms. Repository-level license treatment
does not override the terms of any external dataset or model.
