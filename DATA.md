# DATA.md — dataset provenance, licensing status, and verification

Status: written at Track A4 (2026-07-11). This file is the single source of truth for where
every data file in this repo came from, what may be published, and how to verify integrity.
Machine-readable integrity values live in `data/MANIFEST.json`; verify with
`python scripts/verify_data.py`.

## 1. Datasets at a glance

| File | Records | In git | Origin |
|---|---:|---|---|
| `data/knowledge_base_v1.json` | 8 | yes | Authored in `prepare_data.py` (original text) |
| `data/knowledge_base_v2.json` | 9 | yes | Authored in `prepare_data.py` (original text; simulated KB update) |
| `data/eval_questions.json` | 12 | yes | Authored in `prepare_data.py` (original QA pairs) |
| `data/eval_questions_regression_debug.json` | 4 | yes | Authored debug subset targeting the V2-changed docs |
| `data/large_eval_questions.json` | 500 | **no** | Derived from MS MARCO v2.1 validation split via `scale_up_dataset.py` |
| `data/large_knowledge_base.json` | 498,725 | **no** (~191 MB) | Derived from MS MARCO v2.1 train split via `scale_up_dataset.py` |
| `data/sample_knowledge_base_synthetic.json` | 5 | yes | Synthetic, written for this repo (schema documentation only) |
| `data/sample_eval_questions_synthetic.json` | 3 | yes | Synthetic, written for this repo (schema documentation only) |

## 2. Provenance detail

### 2.1 Small controlled KB + 12-question eval set (original, in-repo)
`knowledge_base_v1.json` (8 docs), `knowledge_base_v2.json` (9 docs — two docs revised, one
added, to simulate a knowledge-base update), `eval_questions.json` (12 QA pairs), and the
4-question regression-debug subset are **authored directly in `prepare_data.py`** — the full
text is checked into this repository and can be diffed against the JSON files. The texts are
original summaries of public technical topics (Transformers, RAG, vector databases, BM25,
etc.), not copies of any external corpus. No external license applies.

### 2.2 Large corpus + 500-question eval set (derived from MS MARCO v2.1)
`scale_up_dataset.py` (checked into this repo) builds both files from **Microsoft MS MARCO
v2.1** (Hugging Face dataset `microsoft/ms_marco`, config `v2.1`):

- `data/large_knowledge_base.json`: streams the **train** split, takes the first 50,000 query
  items, and flattens every passage in each item's `passages.passage_text` into records of the
  form `{"id": "msmarco_<i>_<j>", "title": "Query <i>", "content": <passage text>}`. The run
  used for the saved results produced **498,725 records, ~191 MB** (counted 2026-07-09 from the
  read-only source folder).
- `data/large_eval_questions.json`: streams the **validation** split and keeps the first 500
  items that have answers, as `{"question", "ground_truth", "relevant_doc_ids": []}`.

The 191 MB corpus is **kept out of git permanently** (GitHub hard-rejects files ≥ 100 MB, and
Track A rejected Git LFS: quota/clone friction, no history value). Regeneration is possible by
re-running `scale_up_dataset.py`, with the caveat that Hugging Face streaming order/dataset
revisions may not be byte-stable — the SHA-256 recorded in `data/MANIFEST.json` (see §4) is
the ground truth for "the exact corpus the saved results used".

### 2.3 Synthetic schema samples
`sample_knowledge_base_synthetic.json` and `sample_eval_questions_synthetic.json` are
**invented, clearly-labeled fictional records** written for this repo. They exist so the
corpus/eval schemas are documented in-repo **without republishing MS MARCO content**. They must
never be used as evaluation data.

## 3. Licensing status and publication rules  (B3 live-verified 2026-07-11)

**MS MARCO's terms were verified live on 2026-07-11** via the GitHub API from
`microsoft/msmarco` (the source repository of the official site
https://microsoft.github.io/msmarco/), files `Notice.md` and `README.md`, section
"Terms and Conditions" (default branch as of that date):

> "The MS MARCO and ORCAS datasets are intended for **non-commercial research purposes
> only** to promote advancement in the field of artificial intelligence and related areas,
> and is made available free of charge **without extending any license or other intellectual
> property rights**. The datasets are provided 'as is' without warranty and usage of the data
> has risks since we may not own the underlying rights in the documents."

The historical non-commercial restriction is therefore **confirmed current**. The terms grant
no redistribution rights, and Microsoft notes it may not own the underlying document rights —
so republishing MS MARCO-derived passages/queries in a public portfolio repo is **not
supportable**. Human sign-off remains required only for *how* to remediate before any public
release (rule 2 below); the license question itself is closed.

The binding rules are:

1. **No corpus passages are republished in this repo** — the synthetic samples in §2.3 stand in
   for schema documentation.
2. **`data/large_eval_questions.json` is not tracked in git.** It contains real
   MS MARCO-derived queries/answers when regenerated locally, so public clones must recreate it
   with `scale_up_dataset.py` instead of receiving a republished copy.
3. The full corpus never enters git regardless of license outcome (size policy, §2.2).
4. The small KB/eval files (§2.1) are original in-repo text and carry no such restriction.

## 4. Integrity verification

- `data/MANIFEST.json` records byte size, SHA-256, and top-level record count for every data
  file. Values for the in-git files were computed 2026-07-11 in-session.
- `python scripts/verify_data.py` re-checks every present file against the manifest (exit 1 on
  any mismatch) and is wired into CI at Track A6, which is where it is first exercised in a
  fresh environment.
- The out-of-git corpus is absent on this machine by design; its manifest entry carries
  `records: 498725` (counted 2026-07-09) with `sha256: null`. **First machine that holds the
  file** (Track C0 workstation intake, or the human's Mac source folder) runs
  `python scripts/verify_data.py`, which prints the corpus SHA-256/bytes to record into the
  manifest.

## 5. Claims discipline for anything citing these datasets

- Corpus scale claims ("498,725 records / 191 MB") are labeled *measured 2026-07-09 from the
  source folder*; they become manifest-verified once the corpus SHA-256 is recorded (§4).
- The 12-question eval set and both small KBs are manifest-verified as of 2026-07-11. The
  500-question MS MARCO-derived eval set is tracked only by regeneration instructions and, once
  generated locally, by the manifest's expected record count.
- MS MARCO licensing statements may now cite the verified terms quoted in §3 (source:
  `microsoft/msmarco` `Notice.md`/`README.md`, read via GitHub API 2026-07-11). Claims beyond
  that quote (e.g., about future term changes) remain out of bounds.
