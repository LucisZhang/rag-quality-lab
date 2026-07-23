[English](README.md) | [简体中文](README.zh-CN.md)

# 🔬 RAG Quality Lab

一个完全本地化的评估平台，用于测量、对比和回归测试检索增强生成（RAG）流水线——旨在回答团队通常会跳过的问题：**那次改动是真的让答案质量变好了，还是悄无声息地让它变差了？**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A3E85?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-0F766E?style=flat-square)
![Ollama Gemma 4](https://img.shields.io/badge/Ollama%2FGemma%204-Local%20LLM-111827?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)

## 0. 当前状态一览（2026-07）

本仓库现在包含两个刻意分离的证据层：

- **历史质量基线：** 受控的 12 题 A/B 与回归工作流、2026-04 保存的评判运行，以及在 [`evidence/verified-2026-07/`](evidence/verified-2026-07/README.md) 下于 2026-07 完成的确定性复核。这些小集合指标仍是唯一已发布的答案质量结果。
- **EnterpriseRAG-Bench C2 规模化：** 受版本管理的适配器、清单契约、带文档级标准答案的 130 个可回答问题、后端感知的检索接缝，以及用于处理 **11,309 份合成企业文档**（5,189 条 Confluence 记录加 6,120 条 Jira 记录）的无模型测试。约 88 MB 的适配后知识库保留在 Git 之外，并由经过哈希验证、采用 MIT 许可的 EnterpriseRAG-Bench v1.0.0 切片确定性地重新生成。

C2 证明的是数据与评估基础设施，而不是检索或答案质量。随后的 C3 A/B 时间盒以一条明确的无结果记录告终，而不是来自替代流水线的指标；不应推断任何检索、答案质量、评判或兜底的 C3 结果。§3 中的历史 12 题对比**不能**迁移到 11,309 份文档的范围。这就是本实验室的运行准则：没有指标好过由错误的技术栈产生的指标。

**按读者类型的入门指引：** 在 §6 和 [`DATA.md`](DATA.md) 中查看 11,309 份文档的处理路径；在 §2 和 [`evidence/verified-2026-07/`](evidence/verified-2026-07/README.md) 中评估历史声明；在 §7 中运行本实验室；或者通过 §5、§9、`src/`、`scripts/adapters/` 和 `tests/` 扩展实现。这是一台单机评估实验室，不是已发布或有版本号的软件包，每个结果都带有自己的证据日期和范围。

## 1. 头条发现：一次“无害的”知识库更新降低了质量，而实验室抓住了它

一次知识库更新（V1 → V2：修订两份文档、新增一份——生产 RAG 系统不断吸收的那种例行内容刷新）**使最强流水线在每项质量指标上都出现退化**，其中忠实度（faithfulness）下降最为明显。没有任何代码变更，只有文档变了。

| 指标（流水线 B，12 题集） | V1（基线） | V2（更新后） | 差值 |
| --- | ---: | ---: | ---: |
| 忠实度（Faithfulness） | 0.988 | 0.867 | **−0.121** |
| 答案相关性（Answer Relevancy） | 0.973 | 0.888 | −0.086 |
| 上下文精确率（Context Precision） | 0.913 | 0.850 | −0.063 |
| 上下文召回率（Context Recall） | 0.925 | 0.842 | −0.083 |
| 答案正确性（Answer Correctness） | 0.921 | 0.846 | −0.075 |

逐题分桶：**4 题退化 / 0 题改善 / 8 题稳定**（当任一指标下降超过 0.1 时，该题记为*退化*）。没有回归工具，这种退化会不知不觉地上线给用户；有了它，它就是一个可评审的差异（`results/dashboard_regression/`）。

*证据标签：* 来自 **2026-04 保存运行** 的 LLM 评判分数（本地 `gemma4:e4b` 评判器），于 **2026-07 确定性重新解析**（`scripts/verify_a3.py` → `evidence/verified-2026-07/`）；**全新评判复核尚待进行（工作站）**。4/0/8 分桶和所有计数均已确定性复核。

## 2. 证据状态——引用任何数字前请先阅读本节

本仓库区分两类声明，下面的每个结果表都带有其标签：

- **确定性事实**（数据集计数、文件校验和、索引/延迟计时、保存文件解析）：2026-07 在本副本上通过 `scripts/verify_a3.py` 和 `scripts/verify_data.py` 复核；输出位于 `evidence/verified-2026-07/`。
- **LLM 评判的质量指标**（忠实度、相关性、精确率、召回率、正确性）：产生于 **2026-04 保存运行**，使用通过 Ollama 运行的本地 `gemma4:e4b` 评判器，并在 2026-07 **重新解析——而非重新评判**。2026-07-11 的一次工作站 C0/C1 运行证明了公共克隆、CI、EnterpriseRAG-Bench S1 数据路径以及一次 PyTorch/HF CUDA 嵌入冒烟测试——但 Ollama 本身无法在那里运行（主机的 NVIDIA 驱动 470 早于当前 Ollama 的 CUDA-12 要求），因此在可用硬件上，对 2026-04 基线的**同评判器精确重跑已关闭，结论为不可行**（2026-07-12 决定；笔记本评判路径此前因速度原因已关闭）。全新评判通道被重新限定为通过本仓库的 `ollama|hf` 后端接缝运行的 **Hugging Face 运行时本地评判器（Lane L2）**：它测试*发现*（B 优于 A；V2 更新使 B 退化）能否在独立评判器运行时下复现，其分数将在各自带标签的表格中报告，绝不混入 2026-04 的列。运行记录位于 `evidence/` 下（如存在则为 `evidence/workstation-c0c1-20260711/`；S1 获取记录随附于 `evidence/c2-s1-mac-20260712/`）。来自小型本地评判器的评判绝对分数应被解读为流水线/版本之间的相对信号，而非经过校准的绝对值。

数据集来源、许可状态和完整性校验和见 [`DATA.md`](DATA.md) 与 `data/MANIFEST.json`。

## 3. 流水线 A/B 对比——质量 vs 延迟，两侧均已测量

相同的 12 题评估集、相同的知识库、相同的评判器；只有检索架构不同：

- **流水线 A — NaiveVectorRAG：** Chroma 相似度搜索 → top-k 上下文 → Gemma 4。
- **流水线 B — HybridRerankRAG：** Chroma top-10 + BM25 top-10 → 合并/去重 → 交叉编码器重排序（`cross-encoder/ms-marco-MiniLM-L-6-v2`）→ Gemma 4。

| 指标 | 流水线 A（朴素向量） | 流水线 B（混合+重排序） | 差值 |
| --- | ---: | ---: | ---: |
| 忠实度（Faithfulness） | 0.888 | 0.988 | +0.100 |
| 答案相关性（Answer Relevancy） | 0.804 | 0.973 | +0.169 |
| 上下文精确率（Context Precision） | 0.784 | 0.913 | +0.129 |
| 上下文召回率（Context Recall） | 0.800 | 0.925 | +0.125 |
| 答案正确性（Answer Correctness） | 0.771 | 0.921 | +0.150 |
| **总体均值** | **0.809** | **0.944** | **+0.135（相对 +16.6%）** |

*本表中全部五项指标仅来自 12 题受控集——请将 +16.6% 的提升视为小集合架构信号，而非一般性结论。*

流水线 B 在全部五项指标上胜出；其检索诊断召回率/命中率在该集合上为 1.0。代价见下文 §4：在 50K 文档规模下，检索延迟约为流水线 A 的 4.6 倍。这一对数字——质量提升及其延迟成本——正是本实验室要使之可测量的架构决策。

*证据标签：* 评判指标 = 2026-04 保存运行，2026-07 确定性重新解析（均值 0.8093 → 0.9438，相对提升 +16.6%）；全新评判复核尚待进行（工作站）。源工件：`results/dashboard_comparison/`。

## 4. 规模基准（确定性计时，2026-04 保存运行）

在 MS MARCO 衍生语料上的索引吞吐（嵌入：`nomic-embed-text`，经 Ollama，本地机器）：

| 文档数 | 索引向量数 | 时间（秒） | 每 1K 文档秒数 |
| ---: | ---: | ---: | ---: |
| 1,000 | 1,121 | 13.22 | 13.22 |
| 5,000 | 5,614 | 60.99 | 12.20 |
| 10,000 | 11,290 | 134.85 | 13.49 |
| 50,000 | 56,039 | 691.17 | 13.82 |

在缓存的 50K 文档索引上的检索延迟（100 个确定性查询）：

| 流水线 | P50（毫秒） | P95（毫秒） | P99（毫秒） | 均值（毫秒） |
| --- | ---: | ---: | ---: | ---: |
| 流水线 A | 28.13 | 37.39 | 39.65 | 29.15 |
| 流水线 B | 128.44 | 188.41 | 212.43 | 134.15 |

*证据标签：* 计时为确定性的保存工件（`results/scale_performance/*.csv`，2026-04），2026-07 重新解析。它们刻画的是笔记本级机器；以每 1K 文档约 13.8 秒计算，对完整的 498,725 条记录语料做嵌入，外推约为 1.9 小时——这是进行 GPU 工作站运行的量化论据（外推，而非实测）。

## 5. 架构

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
| Layer 4. Versioned Knowledge Base (data/ - see DATA.md)                          |
|   Controlled: V1 (8 docs) / V2 (9 docs) + 12-question eval set (authored)        |
|   Scale: 498,725 MS MARCO-derived passages (out of git) + 500 eval queries       |
+----------------------------------------------------------------------------------+
```

回归工具（`src/regression_tester.py`）让一个流水线分别对两个知识库版本运行，并将逐题指标差值分桶为改善 / 退化 / 稳定。

## 6. 数据与许可（摘要——完整记录见 `DATA.md`）

- 小型知识库和 12 题评估集是 **在 `prepare_data.py` 中原创撰写的文本**。
- 191 MB / 498,725 条记录的语料和 500 题评估集由 `scale_up_dataset.py` **从 Microsoft MS MARCO v2.1 衍生**。两者在公共仓库中均**保留在 git 之外**；完整性通过 `data/MANIFEST.json` 的预期计数 + `scripts/verify_data.py` 传递。
- **MS MARCO 条款于 2026-07-11 实时核实**：“仅限非商业研究目的……不授予任何许可或其他知识产权”（`microsoft/msmarco` 的 `Notice.md`，通过 GitHub API 阅读）。此处未重新发布任何语料记录；以清晰标注的合成模式样例代替（`data/sample_*_synthetic.json`）。确切引文及本仓库受约束的发布规则见 `DATA.md` §3。

### EnterpriseRAG-Bench S1（Track C 规模化——数据 + 适配器就绪，暂无结果）

规模化轨道将评估迁移到 **EnterpriseRAG-Bench v1.0.0**（MIT，2026-07-11 在 GitHub LICENSE 和 HF 卡片上均已核实）——一个完全合成的企业语料，因此不受 MS MARCO 再分发约束。当前范围为 **S1 = confluence + jira**：11,309 份文档和一个带文档级标准答案（`expected_doc_ids`，`dsid_*` 粒度）的 **130 题可回答池**，这使得在任何评判运行之前即可进行**无评判器的确定性检索评估**（`scripts/run_s1_retrieval_ab.py`）。

- 获取完整性：两台机器独立下载了六个发布文件，其 SHA-256 校验和完全一致（`evidence/c2-s1-mac-20260712/`）。
- 适配器（`scripts/adapters/`）确定性地将切片 → 本实验室的知识库模式、将 `questions.jsonl` → 评估模式；`data/MANIFEST.json` 承载两份输出的完整性（约 88 MB 的适配知识库保留在 git 之外，并可逐字节一致地重新生成）。
- **如实说明状态：** 目前尚无任何 S1 检索或质量结果——第一批数字将来自计划中的 GPU 工作站运行（C3）。此处没有任何内容声称相反的情况。

## 7. 快速开始

### 前置条件

- Python 3.11（固定于 `.python-version`）
- [Ollama](https://ollama.com/)
- 足够的空闲磁盘，用于本地模型、Chroma 持久化和基准工件

### 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt   # exact pinned versions
# (requirements.txt is the smaller top-level spec; the lockfile is the reproducible path)
```

锁定文件如何产生以及冒烟测试记录见 `docs/A2_ENVIRONMENT.md`。

### 拉取本地模型

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama serve   # if not already running
```

### 模型后端（默认 `ollama`，可选 `hf`）

不做任何配置时，一切使用上述本地 Ollama 模型——行为不变。在无法运行 Ollama 的机器上（例如 NVIDIA 驱动早于当前 Ollama CUDA-12 要求的 GPU 主机），可按组件选择 Hugging Face 后端：

```bash
export RAG_MODEL_BACKEND=hf              # both components, or use per-component:
export RAG_EMBEDDING_BACKEND=hf          #   embeddings only (retrieval-only runs)
export RAG_HF_LLM_MODEL=<org/model-id>       # required for the hf LLM
export RAG_HF_EMBEDDING_MODEL=<org/model-id> # required for hf embeddings
# optional: RAG_HF_DEVICE=cuda:0  RAG_HF_MAX_NEW_TOKENS=512  RAG_HF_NORMALIZE_EMBEDDINGS=1
```

刻意**不提供默认的 HF 模型 id**：首次下载前请核实模型卡片（许可、大小/显存），然后再导出 id。`hf` 后端需要完整的锁定文件环境（`langchain-huggingface` 在其中固定）；CI 从不触及它。

### 数据

小型受控数据集随附于 `data/`。随时可验证完整性：

```bash
python scripts/verify_data.py
```

预期结果：脚本对照 `data/MANIFEST.json` 重新校验每个现存数据文件，并在任何不匹配时以非零码退出（`DATA.md` §4），因此干净退出意味着受版本管理的数据与清单一致。

大型语料不在 git 中；可重新生成（需要网络，流式读取 MS MARCO v2.1）：

```bash
python scale_up_dataset.py
```

### 启动

```bash
streamlit run app.py
```

预期结果：Streamlit 启动并在本地端口提供仪表板外壳。记录在案的 2026-07-10 无头冒烟测试（`docs/A2_ENVIRONMENT.md`）绑定了 `127.0.0.1:8521` 并返回 `HTTP/1.1 200 OK`。

### 复核已保存的证据

```bash
python scripts/verify_a3.py   # deterministic re-checks -> evidence/verified-2026-07/
```

### 运行单元测试（无需模型）

```bash
pip install -r requirements-ci.txt   # lightweight: no torch/chromadb/ollama
pytest                               # model-free logic; all LLM calls mocked
ruff check .                         # lint
python scripts/verify_data.py        # data integrity vs data/MANIFEST.json
```

测试覆盖无模型核心——分块、混合 BM25/稠密向量的合并-去重-重排序、检索指标和回归差异比对——以脚本化假件替代向量库、交叉编码器和评判 LLM（`tests/dependency_stubs.py` 仅在完整环境缺失时替代重依赖导入）。GitHub Actions（`.github/workflows/ci.yml`）在推送到 `main` 及拉取请求时运行相同的四条命令外加 `verify_a3.py` 的确定性部分；CI 从不执行模型推理。

## 8. 使用指南

**标签页 1 — 交互式查询：** 让一个问题经由流水线 A 或 B 运行；在生成的答案旁检查检索到的文档（id、标题、分数）。

**标签页 2 — 流水线 A/B 对比：** 在同一问题集上评估两条流水线（数量滑块 2–12）；输出雷达图、分组柱状图、逐题分数表和总结结论。工件落在 `results/dashboard_comparison/`。

**标签页 3 — 回归测试：** 让一条流水线跨 `knowledge_base_v1.json` 与 `knowledge_base_v2.json` 对比。显示指标差值表、改善/退化/稳定计数、逐题状态卡片（V1 vs V2 答案、检索到的文档 id、逐指标差值）以及知识库变更摘要（新增/修改/删除/未变的文档）。阈值：任一指标 < −0.1 记为退化，任一指标 > +0.1 记为改善，否则为稳定。工件落在 `results/dashboard_regression/`。

**标签页 4 — 规模与性能：** 在 1K/5K/10K/50K 语料子集上的索引基准，以及一个纯检索延迟基准（对缓存的 50K 索引运行 100 个确定性查询）。工件落在 `results/scale_performance/`。

## 9. 用自定义流水线扩展

各流水线共享同一契约（`src/rag_pipelines.py`）：

```python
def query(self, query: str, k: int = 4) -> dict[str, Any]:
    return {
        "question": query,
        "answer": answer,
        "contexts": [doc.page_content for doc in retrieved_docs],
        "retrieved_doc_ids": retrieved_doc_ids,
    }
```

继承 `BaseRAGPipeline` 并实现：

```python
def retrieve(self, query: str, k: int = 4) -> list[Document]: ...
```

要添加流水线 C：实现该类，在 `app.py` 中添加一个带缓存的工厂，在 `PIPELINE_CLASS_MAP` 和 `PIPELINE_APP_CONFIG` 中注册，并更新流水线选择器。（A/B 对比流程目前硬编码为两条流水线；泛化 `compute_comparison_evaluation()` 及其图表是已知的扩展点。）

`RAGEvaluator.load_eval_dataset()` 要求每个评估条目具有 `question`、`ground_truth` 和 `relevant_doc_ids`（后者可以为空，如重新生成的 `data/large_eval_questions.json` 所示）。数据文件模式在 `DATA.md` 和合成样例中有带示例的文档说明。

## 10. 项目结构

```text
rag-quality-lab/
├── app.py                        # Streamlit dashboard (four modes)
├── src/
│   ├── utils.py                  # Ollama clients, JSON loading, chunking, Chroma creation
│   ├── rag_pipelines.py          # BaseRAGPipeline, NaiveVectorRAG, HybridRerankRAG
│   ├── evaluation_engine.py      # RAGEvaluator (RAGAS + local-judge fallback)
│   └── regression_tester.py      # RegressionTester and report generation
├── data/
│   ├── knowledge_base_v1.json    # 8-doc baseline KB (authored)
│   ├── knowledge_base_v2.json    # 9-doc updated KB (authored)
│   ├── eval_questions.json       # 12-question controlled eval set (authored)
│   ├── eval_questions_regression_debug.json  # 4-question debug subset
│   ├── eval_questions_enterpriserag_s1.json  # 130-question S1 pool (MIT, adapted)
│   ├── sample_*_synthetic.json   # labeled synthetic schema samples
│   └── MANIFEST.json             # checksums / sizes / record counts
├── results/                      # saved 2026-04 experiment artifacts (CSV/JSON)
├── evidence/                     # verification outputs + acquisition records
├── scripts/
│   ├── verify_a3.py              # deterministic re-checks of saved artifacts
│   ├── verify_data.py            # data integrity vs MANIFEST.json
│   ├── run_s1_retrieval_ab.py    # judge-free S1 retrieval A/B (deterministic)
│   ├── adapters/                 # EnterpriseRAG-Bench S1 -> lab schema converters
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

（完整语料、Chroma 存储、虚拟环境和生成的运行时工件按设计被 gitignore——见 `.gitignore` 和 `docs/A1_COPY_NOTES.md`。）

## 11. 技术栈

| 组件 | 技术 | 版本（锁定文件） | 用途 |
| --- | --- | --- | --- |
| 语言 | Python | 3.11 | 核心实现语言 |
| 仪表板 | Streamlit | 1.45.1 | 交互式本地 UI |
| RAG 框架 | LangChain | 0.3.25 | 提示、文档、编排原语 |
| LangChain 集成 | `langchain-community` | 0.3.24 | 社区集成 |
| Ollama 集成 | `langchain-ollama` | 0.3.3 | 本地 LLM 与嵌入访问 |
| 向量库集成 | `langchain-chroma` | 0.2.6 | Chroma 的 LangChain 封装 |
| 向量数据库 | ChromaDB | 1.5.7 | 持久化向量检索 |
| 评估框架 | RAGAS | 0.2.15 | 自动化 RAG 评估 |
| 数据集加载器 | `datasets` | 3.6.0 | 流式 MS MARCO 摄取 |
| 词法检索 | `rank-bm25` | 0.2.2 | BM25 候选检索 |
| 重排序器 | `sentence-transformers` | 4.1.0 | 交叉编码器重排序 |
| 可视化 | Plotly | 6.1.2 | 图表与基准绘图 |
| 数据分析 | Pandas | 2.2.3 | 表格结果处理 |
| 数值工具 | NumPy | 1.26.4 | 数值支持 |
| 亦固定 | `langchain-huggingface`、`openpyxl` | 0.2.0、3.1.5 | HF 嵌入封装、Excel 导出 |

默认模型配置（`src/utils.py`，`ollama` 后端）：LLM `gemma4:e4b`，嵌入 `nomic-embed-text`，Ollama 端点 `http://127.0.0.1:11434`；`hf` 后端（§7）将这些替换为由环境变量命名的 Hugging Face 模型。流水线 B 在首次运行时还会下载 `cross-encoder/ms-marco-MiniLM-L-6-v2`。

## 12. 局限性（如实说明范围）

1. **评判器保真度与时效性。** 质量指标来自 2026-04 保存运行中的小型本地评判器（经 Ollama 的 `gemma4:e4b`）。它们在 2026-07 被确定性地重新解析，但**尚未重新评判**——而且同评判器精确重跑现已**关闭，结论为不可行**：笔记本慢得不切实际（1 题探针，2026-07），GPU 工作站的 NVIDIA 驱动 470 无法运行当前的 Ollama（2026-07-11 运行；该路径于 2026-07-12 退役）。替代方案是一条全新的 **HF 运行时评判通道（Lane L2）**，测试这些发现能否在独立评判器运行时下复现；在其运行之前，请谨慎对待绝对分数——流水线与知识库版本之间的差值才是有意义的信号。
2. **受控评估规模。** 高保真 A/B 与回归工作流运行在小型手工撰写的 12 题集上；500 题集驱动规模基准，而非评判质量运行。
3. **单机范围。** 一切都在一台本地机器上运行；规模上限是笔记本上的 50K 文档索引。更大规模的运行是计划中的工作站轨道，有其自身的证据纪律。
4. **许可门控的数据发布。** MS MARCO 的已核实条款（仅限非商业研究，无再分发权利——`DATA.md` §3）意味着 MS MARCO 衍生内容不进入任何公开发布；仓库改用合成样例来记录模式。
5. **尚无 UI 截图。** 仪表板截图将从实际会话中截取并添加；在此之前不重建或伪造任何截图。

## 13. 参考文献

- *Corrective Retrieval Augmented Generation*. arXiv, 2024. <https://arxiv.org/abs/2401.15884>
- *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv, 2023. <https://arxiv.org/abs/2309.15217>
- *Okapi at TREC-3*. NIST TREC-3 Proceedings, 1994. <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>
- *Passage Re-ranking with BERT*. arXiv, 2019. <https://arxiv.org/abs/1901.04085>
- *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv, 2020. <https://arxiv.org/abs/2002.10957>
- *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv, 2016. <https://arxiv.org/abs/1611.09268>

## 14. 权利

本仓库目前未授予任何开源许可；保留所有权利。数据集许可单独且严格地受其自身条款约束：本仓库所受的 MS MARCO 非商业、禁止再分发条款见 `DATA.md`；已跟踪的 C2 数据与评估基础设施所使用的 EnterpriseRAG-Bench 切片采用 MIT 许可。
