# C3 deterministic S1 retrieval A/B - timebox record (2026-07-12)

Status: **timebox expired / blocked before execution**. No C3 metrics were produced.
The portfolio page must remain on the C2 fallback and must not show an S1 A/B table.

## Intended run

- Runner: `scripts/run_s1_retrieval_ab.py`
- Scope: all 130 S1-answerable questions over 11,309 documents
- A: the lab's existing `NaiveVectorRAG` (Chroma dense retrieval)
- B: the lab's existing `HybridRerankRAG` (Chroma dense top-10 + BM25 top-10,
  content dedupe, then `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking)
- Final `k`: 4
- Scoring: deterministic document-level precision, recall, and hit at deduplicated
  `dsid_*` granularity; no LLM judge, generation, API key, or paid call

## Preflight result

The local Ollama service was reachable and already held `nomic-embed-text:latest`.
However, the project `.venv` did not contain `chromadb`, `langchain-chroma`,
`langchain-ollama`, `sentence-transformers`, `torch`, or `transformers`. The required
cross-encoder reranker was not in the local Hugging Face cache. A local pip-cache and
alternate-environment search found no reusable installation of that stack.

Completing the real hybrid run therefore requires installing the pinned heavy stack
(including PyTorch) and downloading the cross-encoder. That is outside this task's
offline, no-multi-GB-install timebox. Replacing the pipeline with a toy lexical or
otherwise easier algorithm would invalidate the requested A/B, so no fallback metrics
were generated. Exact machine-readable preflight facts and local model digests are in
`dependency-preflight.json`.

## Justified C3 runner improvements

The runner was improved without changing either retrieval algorithm. A completed future
run now records:

- actual document and question counts;
- SHA-256, byte size, and record count for both adapted datasets;
- SHA-256 and byte size for both adapter scripts;
- stable definitions of both real pipelines, candidate depths, final `k`, chunking, and
  reranker model id;
- Python/platform and relevant package versions, plus active embedding/LLM model ids;
- per-pipeline metrics and per-question outputs as before;
- a separate `MANIFEST.json` hashing every result CSV, summary JSON, and comparison JSON.

Focused model-free tests cover the pipeline-definition contract and output-manifest
hashes. This record deliberately contains no `results/` artifact and no portfolio metric
claim.
