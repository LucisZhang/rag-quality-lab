# A3 Fresh Run Attempt

Started: 2026-07-10T09:34:25Z
Mode: `scripts/verify_a3.py --mode fresh --max-questions 1`

Outcome: stopped manually after the 1-question smoke exceeded six minutes on the local Mac.

What completed before stop:
- NaiveVectorRAG loaded 9 documents, chunked them into 27 pieces, built a Chroma index, generated an answer, and completed fallback LLM scoring for 1 question.
- HybridRerankRAG loaded 9 documents, chunked them into 27 pieces, built a Chroma index, built the BM25 index, loaded the cross-encoder reranker, and generated an answer for 1 question.

Where it stalled:
- HybridRerankRAG reached fallback local-LLM judge scoring.
- The local `gemma4:e4b` judge path was too slow for a full 12-question A/B plus regression run on this Mac in the current setup.

Decision:
- Do not claim fresh 12-question re-verification yet.
- Keep the deterministic A3 checks as completed evidence.
- Full fresh A/B + regression should be re-run after either:
  - moving the judge workload to the workstation path,
  - selecting a cheaper/faster judge configuration, or
  - reducing the fresh-run scope and labeling it as a smoke test instead of full A3 re-verification.

Generated runtime Chroma files are ignored by git because they are intermediate caches.
