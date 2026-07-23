# A3 Verification Evidence - 2026-07

Status: PARTIAL.

Deterministic checks completed on 2026-07-10. Full fresh 12-question A/B and V1->V2
local-judge regression were attempted only as a 1-question smoke and stopped because the local
`gemma4:e4b` fallback judge path was too slow on this Mac.

## Completed Checks

| Check | Result |
|---|---|
| `data/knowledge_base_v1.json` count | 8 documents |
| `data/knowledge_base_v2.json` count | 9 documents |
| `data/eval_questions.json` count | 12 questions |
| `data/large_eval_questions.json` count | 500 questions when regenerated locally; absent by design in public clones |
| Full large corpus in clean copy | Not present; intentionally excluded from git |
| Eval dataset schema load | 12 valid rows |
| Ollama model availability | `gemma4:e4b` and `nomic-embed-text` available via local Ollama API |

## Saved-Run Metrics Re-Parsed

These values were re-parsed from committed saved artifacts in `results/`; they are not new
LLM-judge scores.

| Metric | Saved artifact value | 2026-07 deterministic re-parse |
|---|---:|---:|
| Pipeline A five-metric mean | 0.8093 | 0.8093 |
| Pipeline B five-metric mean | 0.9438 | 0.9438 |
| Relative lift | +16.6% | +16.6% |
| Regression degraded questions | 4 | 4 |
| Regression improved questions | 0 | 0 |
| Regression stable questions | 8 | 8 |
| Max indexing benchmark size | 50,000 docs | 50,000 docs |

## Fresh-Run Status

Fresh run command attempted:

```bash
/Users/hsiangkuochang/rag-quality-lab/venv/bin/python scripts/verify_a3.py --mode fresh --max-questions 1
```

The smoke reached HybridRerankRAG answer generation, including Chroma indexing, BM25, and
cross-encoder reranker load. It stalled during fallback local-LLM judge scoring and was stopped
after more than six minutes.

Conclusion: do not claim fresh 12-question re-verification yet. The full fresh A/B + regression
run should move to a faster workstation or a faster/capped judge configuration.

Follow-up: `evidence/workstation-c0c1-20260711/` records the first workstation C0/C1 attempt.
That run completed CI-equivalent checks, EnterpriseRAG-Bench S1 acquisition/counts, and a
PyTorch/HF CUDA embedding smoke, but the original Ollama-judged fresh A3 path remained blocked
by Ollama/driver compatibility.
