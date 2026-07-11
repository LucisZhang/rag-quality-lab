# A2 Environment And Smoke-Test Notes

Created: 2026-07-10

## Python Pin

- Required Python line: 3.11
- Local interpreter used for this checkpoint: Python 3.11.15
- Pin file: `.python-version`

## Dependency Lock

- Human-readable dependency list: `requirements.txt`
- Reproducibility lockfile generated from the previously working project venv:
  `requirements-lock-py311.txt`
- The lockfile was generated with:

```bash
/Users/hsiangkuochang/rag-quality-lab/venv/bin/python -m pip freeze > requirements-lock-py311.txt
```

The source venv is not copied into this repo and remains excluded from git.

## Fresh Install Path

Use this path on a clean machine or a larger workstation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt
```

For lighter local work, `requirements.txt` remains available as the smaller top-level dependency spec.

## Headless Streamlit Smoke Test

To conserve local disk, the A2 smoke test uses the original already-working Python 3.11 venv
as the interpreter while running the clean-copy app code from this repo:

```bash
/Users/hsiangkuochang/rag-quality-lab/venv/bin/python -m streamlit run app.py \
  --server.headless true \
  --server.address 127.0.0.1 \
  --server.port 8521
```

Expected result: Streamlit starts, binds to `127.0.0.1:8521`, and serves the app shell without
copying large vector stores or the full 191 MB corpus into git.

Actual result on 2026-07-10:
- Streamlit started successfully on `http://127.0.0.1:8521`.
- `curl -I http://127.0.0.1:8521` returned `HTTP/1.1 200 OK`.
- The response body served the Streamlit HTML app shell.
- The temporary server was stopped after the probe.
