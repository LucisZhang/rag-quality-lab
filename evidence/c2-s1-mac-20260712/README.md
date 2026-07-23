# C2 S1 Acquisition — Mac Mirror Integrity Record (2026-07-12)

Status: C2-1 complete. The Mac independently downloaded the EnterpriseRAG-Bench S1 subset
and verified it byte-identical to the 2026-07-11 workstation acquisition. Two machines,
two independent downloads from the upstream release, matching SHA-256 on all six files.

## Source

- Dataset: EnterpriseRAG-Bench v1.0.0 (onyx-dot-app), synthetic enterprise corpus
  (LLM-generated "Redwood Inference" company simulation).
- License: MIT — verified 2026-07-11 on both the GitHub repo LICENSE and the Hugging Face
  dataset card (`onyx-dot-app/EnterpriseRAG-Bench`).
- Download date (this mirror): 2026-07-12, from
  `https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/`
- Files: `confluence_slice_0001.zip`, `confluence_slice_0002.zip`, `jira_slice_0001.zip`,
  `jira_slice_0002.zip`, `questions.jsonl`, `extra_questions.jsonl`
- Local mirror path (gitignored, regenerate by re-downloading):
  `data/enterpriserag-bench/v1.0.0/`

## SHA-256 cross-check (Mac 2026-07-12 vs workstation 2026-07-11)

All six files matched the workstation acquisition record
(`SHA256SUMS.workstation-20260711.txt`) exactly — `shasum -a 256 -c` reported OK for 6/6:

```text
a79320d9c11b58c904ddc444f4b5c374ea1d9e6e3075974bcaea294c2a7e2eb5  confluence_slice_0001.zip
ecb4e395f710f0eef8332b1e27182c00223da847945a010c10078b3e0c693024  confluence_slice_0002.zip
26e23e5ade467512433e0fc012b30ea14b079dde11ad6f93cf380eed1bd96807  extra_questions.jsonl
71d6ca9c0482e091382085089750a730ca1cb8336ee1cc06e6f089772f0a9747  jira_slice_0001.zip
345e37bd3b6f473eba43115014ea6b7df3de6d1dcf72dfc2f39e1b93cfaa1a14  jira_slice_0002.zip
f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905  questions.jsonl
```

A copy of this list is `SHA256SUMS.mac-20260712.txt` in this directory.

## Extraction counts (Mac, matching the workstation record)

| Item | Count |
|---|---:|
| Confluence `.txt` files | 5,189 |
| Jira `.txt` files | 6,120 |
| Total S1 documents | 11,309 |
| `questions.jsonl` records | 500 |
| S1-answerable pool (`source_types` ⊆ {confluence, jira}) | 130 |
| `extra_questions.jsonl` records (all type `metadata`; separate from the 500) | 100 |

Question-type counts derived from `questions.jsonl` itself (the HF card and GitHub
quickstart disagree on per-category counts, so the file is authoritative): basic 175,
semantic 125, intra_document_reasoning 40, project_related 40, constrained 30,
conflicting_info 20, completeness 20, miscellaneous 20, high_level 10, info_not_found 20.

## Schema facts that drive the C2 adapters

- Every document filename is `dsid_<32 hex>__<slug>.txt`; the `dsid_*` prefix is the
  document id that `questions.jsonl` references via `expected_doc_ids`.
- Question records carry: `question_id`, `question_type`, `source_types`, `question`,
  `expected_doc_ids`, `gold_answer`, `answer_facts`. Document-level ground truth therefore
  exists, enabling judge-free deterministic retrieval evaluation.
- All 130 S1-pool questions' `expected_doc_ids` resolve within the S1 corpus; 30 of the
  130 expect multiple documents; none are `info_not_found`; none have empty ids.

## Findings (recorded, not hidden)

1. `dsid_feb1e9063ebb4947bb4f935393c01f0f` appears in BOTH confluence and jira with
   different content (an incident-review page and its related support ticket sharing one
   dsid). Not referenced by any of the 500 questions.
2. `dsid_6df52fdb96ae4edcb76464738bca3340` appears TWICE within jira with different
   content (two versions of ticket INT-7832). It IS referenced by S1 question `qst_0413`,
   whose `expected_doc_ids` lists it twice.
3. Adapter policy adopted: the knowledge-base document id IS the dsid; multiple files may
   share a dsid (they are facets/versions of one logical document); retrieval scoring is
   at dsid granularity; per-question expected ids are deduplicated.
