# Workstation C0/C1 Evidence - 2026-07-11

Status: partial C0/C1 run. The workstation proved the public repo, CI, EnterpriseRAG-Bench S1
data path, and a GPU-backed PyTorch/HF embedding smoke. The original Ollama-based fresh judged
A3 run did not complete because the current Ollama runtime was incompatible with the
workstation's NVIDIA driver.

## Source

- Workstation run timestamp: 2026-07-11T15:21:05Z
- Evidence tarball on Mac: `/Users/hsiangkuochang/Downloads/rag-c0c1-evidence-20260711.tgz`
- Tarball SHA-256:
  `045abdb4bb510ad5956093187bdbe42c159fce24368f412b773679eb971539cf`
- Public repo cloned on workstation: `https://github.com/LucisZhang/rag-quality-lab`
- Workstation repo path: `/nfs/dataset-ofs-algo/zhangxiangguo/repos/rag-quality-lab`

## Results

| Area | Result |
|---|---|
| Steps 1-5 | PASS |
| CI-equivalent | Ruff clean; 44 tests passed; `verify_data.py` passed; deterministic A3 passed |
| Ollama setup | BLOCKED |
| Fresh Ollama-judged A3 | SKIPPED/BLOCKED |
| Large MS MARCO KB regeneration | SKIPPED |
| EnterpriseRAG-Bench S1 download | PASS |
| EnterpriseRAG-Bench S1 extraction/counts | PASS |
| PyTorch/HF CUDA embedding smoke | PASS |

## EnterpriseRAG-Bench S1 Counts

- Confluence files: 5,189
- Jira files: 6,120
- Questions: 500
- S1-answerable question pool: 130
- Full `all_documents.zip` / full 500K corpus: not downloaded

## Embedding Smoke

- Smoke type: PyTorch/HF fallback
- Model: `BAAI/bge-base-en-v1.5`
- Documents: 1,000
- Device: CUDA / NVIDIA RTX A6000
- Seconds per 1K docs: 15.73
- Peak CUDA memory: 1.011 GB
- Note: this is not directly comparable to the Mac `nomic-embed-text`/Ollama baseline. It proves
  the workstation can execute a GPU-backed embedding workload through PyTorch.

## Ollama Blocker

The original plan required Ollama for `gemma4:e4b` judged A3 and `nomic-embed-text` embedding.
That path was blocked:

- The current Ollama Linux package extracted successfully.
- `gemma4:e4b` and `nomic-embed-text` pulled successfully.
- Ollama generated shared-home key files under `/home/luban/.ollama/`; those files were moved to
  quarantine on the workstation and were not included in this evidence directory.
- Ollama reported the host NVIDIA driver 470 was below its required driver level and selected
  CPU with 0 B VRAM.
- The Ollama server did not remain reachable on port 11434.

Therefore, the fresh local-judge A3 run remains open. Future options are: use a runtime/host with
a newer NVIDIA driver, find and pin a compatible older Ollama version, or rewrite the judged path
to use a PyTorch/HF local model stack.

## Files

- `final-report.txt` - normalized human/Codex final report from the workstation run.
- `deterministic-checks.workstation.json` - deterministic A3 re-parse produced on the
  workstation/public clone.
- `logs/c0-ci-equiv.log` - CI-equivalent command output.
- `logs/c1-counts.log` - EnterpriseRAG-Bench S1 counts and question-type breakdown.
- `logs/c1-embed-smoke-pytorch-hf-20260711-231549.log` - PyTorch/HF embedding smoke output.
- `enterpriserag-bench-v1.0.0/PROVENANCE.txt` - dataset provenance.
- `enterpriserag-bench-v1.0.0/SHA256SUMS.workstation-20260711.txt` - downloaded S1 checksums.
- `enterpriserag-bench-v1.0.0/embed_smoke_result.json` - machine-readable embedding smoke result.
