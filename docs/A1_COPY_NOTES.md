# A1 Clean Copy Notes

Created: 2026-07-10

Source: `/Users/hsiangkuochang/rag-quality-lab` (read-only reference)
Destination: `/Users/hsiangkuochang/rag-quality-lab-portfolio`

Included:
- Application/code files: `app.py`, `src/`, `tools/`, `prepare_data.py`, `scale_up_dataset.py`
- Project docs and dependency spec: `README.md`, `requirements.txt`
- Small controlled datasets and eval files under `data/`
- Saved result artifacts under `results/`

Excluded from the initial working copy:
- Local Python/runtime caches: `venv/`, `.venv/`, `__pycache__/`, `node_modules`
- Vector stores and probes: `chroma_db*/`, `tmp_chroma_*`
- Duplicate source snapshot: `Code/`
- Generated outputs and benchmark caches: `outputs/`, `benchmark_artifacts/`
- Generated regression vector stores: `results/regression_runtime/`, `results/regression_smoke/`
- Full large corpus: `data/large_knowledge_base.json`

Pre-commit checks:
- Large-file scan: no files >= 100 MB in this clean copy.
- Targeted credential scan: no matches for common API key, GitHub token, AWS key, or private-key patterns.

Known deferred data item:
- The source full corpus `data/large_knowledge_base.json` is about 191 MB and must stay out of git.
- Track A4 will replace it with a publishable manifest/sample/checksum flow after exact source/license text is written down.

Next task: A2 pin Python 3.11, produce a lockfile/quickstart, and run a headless Streamlit smoke test.
