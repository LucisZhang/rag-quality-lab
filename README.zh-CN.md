[English](README.md) | [简体中文](README.zh-CN.md)

# 🔬 RAG Quality Lab

一个完全本地运行的评估实验室，用于测量、对比和回归测试检索增强生成（Retrieval-Augmented Generation）流水线——为回答团队最常跳过的问题而构建：**那次改动到底让答案质量变好了，还是悄无声息地让它变差了？**

本实验室曾具体地回答过这个问题一次：一次常规的知识库更新——生产 RAG 系统持续不断地吸收的那类更新——**在每一项质量指标上都使最强流水线的表现下降**，而只有回归测试框架让这一点暴露出来（§1）。正是这一发现，促使本仓库从一个受控的 12 题测试台，发展为针对 11,309 篇文档的合成企业语料库、有版本追踪、经过测试的评估基础设施（§3）。

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A3E85?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-0F766E?style=flat-square)
![Ollama Gemma 4](https://img.shields.io/badge/Ollama%2FGemma%204-Local%20LLM-111827?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)

**阅读路径：** 核心发现及其确切适用范围见 §1–§2；企业级规模的基础设施见 §3；架构见 §4；安装与验证命令见 §5；证据规则与数据出处见 §6 和 [`DATA.md`](DATA.md)；扩展点见 §8；仓库结构图见 §9。这是一个可工作的单机评估实验室，不是已发布或带版本号的软件包，下文每一项结果都附带各自的证据日期和适用范围。

## 1. 核心发现：一次“无害”的知识库更新降低了质量，而实验室捕捉到了它

一次知识库更新（V1 → V2：修订两篇文档，新增一篇）**使最强流水线在每一项质量指标上下降**，其中忠实度（faithfulness）下降最为明显。代码没有任何改动，只有文档变了。

| 指标（流水线 B，12 题集） | V1（基线） | V2（更新后） | 差值 |
| --- | ---: | ---: | ---: |
| Faithfulness（忠实度） | 0.988 | 0.867 | **−0.121** |
| Answer Relevancy（答案相关性） | 0.973 | 0.888 | −0.086 |
| Context Precision（上下文精确率） | 0.913 | 0.850 | −0.063 |
| Context Recall（上下文召回率） | 0.925 | 0.842 | −0.083 |
| Answer Correctness（答案正确性） | 0.921 | 0.846 | −0.075 |

逐题分桶：**4 题下降 / 0 题改善 / 8 题稳定**（当某题的任一指标下降超过 0.1 时，该题记为*下降*）。没有回归测试框架，这样的退化会在无人察觉的情况下发布给用户；有了它，这就是一份可审查的差异报告（`results/dashboard_regression/`）。

*证据：* 由 LLM 评判的分数来自**已保存的 2026-04 运行**（通过 Ollama 使用本地 `gemma4:e4b` 评判模型），并由 `scripts/verify_a3.py` 于 **2026-07 以确定性方式重新解析**到 `evidence/verified-2026-07/`。4/0/8 分桶和所有计数均已通过确定性方式重新验证；评判分数本身未被重新评判——适用范围与注意事项见 §6。这些 12 题指标仅适用于受控题集。

## 2. 为什么“最强流水线”是有证据支撑的说法：A/B 对比

同一 12 题评估集、同一知识库、同一评判模型；只有检索架构不同：

- **流水线 A — NaiveVectorRAG：** Chroma 相似度搜索 → top-k 上下文 → Gemma 4。
- **流水线 B — HybridRerankRAG：** Chroma top-10 + BM25 top-10 → 合并/去重 → 交叉编码器重排序（`cross-encoder/ms-marco-MiniLM-L-6-v2`）→ Gemma 4。

| 指标 | 流水线 A（朴素向量） | 流水线 B（混合+重排序） | 差值 |
| --- | ---: | ---: | ---: |
| Faithfulness（忠实度） | 0.888 | 0.988 | +0.100 |
| Answer Relevancy（答案相关性） | 0.804 | 0.973 | +0.169 |
| Context Precision（上下文精确率） | 0.784 | 0.913 | +0.129 |
| Context Recall（上下文召回率） | 0.800 | 0.925 | +0.125 |
| Answer Correctness（答案正确性） | 0.771 | 0.921 | +0.150 |
| **总体均值** | **0.809** | **0.944** | **+0.135（相对提升 +16.6%）** |

流水线 B 在全部五项指标上胜出（其在该题集上的检索召回率/命中率为 1.0）——请将 +16.6% 的提升视为小样本集上的架构信号，而非普适结论。质量只是决策的一面；测试框架同样测量了成本的一面：

| 流水线 | P50（毫秒） | P95（毫秒） | P99（毫秒） | 均值（毫秒） |
| --- | ---: | ---: | ---: | ---: |
| 流水线 A | 28.13 | 37.39 | 39.65 | 29.15 |
| 流水线 B | 128.44 | 188.41 | 212.43 | 134.15 |

在缓存的 50K 文档索引上的检索延迟（100 个确定性查询）：流水线 B 为获得质量提升，付出了约 4.6 倍于流水线 A 的延迟。在同一 MS MARCO 衍生语料上的索引吞吐（通过 Ollama 使用 `nomic-embed-text` 嵌入，笔记本级机器）：

| 文档数 | 已索引向量数 | 时间（秒） | 每千文档秒数 |
| ---: | ---: | ---: | ---: |
| 1,000 | 1,121 | 13.22 | 13.22 |
| 5,000 | 5,614 | 60.99 | 12.20 |
| 10,000 | 11,290 | 134.85 | 13.49 |
| 50,000 | 56,039 | 691.17 | 13.82 |

*证据：* 评判指标 = 已保存的 2026-04 运行，2026-07 以确定性方式重新解析（均值 0.8093 → 0.9438，相对提升 +16.6%；产物位于 `results/dashboard_comparison/`）。延迟/索引计时为确定性的已保存产物（`results/scale_performance/*.csv`，2026-04），2026-07 重新解析。按每千文档约 13.8 秒计算，对完整的 498,725 条记录的语料做嵌入，在该硬件上外推约为 1.9 小时——这是外推估算，而非实测。

## 3. 从 12 道题到 11,309 篇企业文档

12 题测试框架在一个小到可以人工逐篇审计的语料上证明了论点：RAG 的改动需要有证据支撑的质量门禁，因为看似无害的改动会把东西弄坏。随之而来的问题是：同一套评估生命周期——版本化数据集、清单校验完整性、确定性重验证、无模型 CI——在大到无法人工查看的企业形态语料上是否依然成立。这一规模化工作现已作为可工作的基础设施在本仓库中得到追踪：

- **语料库：** **EnterpriseRAG-Bench v1.0.0**（MIT 许可，已于 2026-07-11 在 GitHub LICENSE 和 HF 数据集卡片两处核实）——一个完全合成的企业语料库，不受 MS MARCO 再分发限制的约束。当前范围：confluence + jira 两个切片，**5,189 + 6,120 = 11,309 篇文档**，以及一个 **130 题的可回答问题池**，携带文档级标准答案（`expected_doc_ids`，粒度为 `dsid_*`）。
- **获取完整性：** 两台机器独立下载了六个发布文件；SHA-256 校验和 6/6 全部一致（`evidence/c2-s1-mac-20260712/`、`evidence/workstation-c0c1-20260711/`）。
- **确定性适配器**（`scripts/adapters/`）：将切片转换为本实验室的知识库 schema，并将 `questions.jsonl` 转换为评估 schema。约 88 MB 的适配后知识库不入 git，且可**逐字节一致地**重新生成——`data/MANIFEST.json` 记录了任何机器都必须复现的 SHA-256。原始数据的怪异之处（重复的 `dsid` 编号）如实记录在案，不做静默修补（`evidence/c2-s1-mac-20260712/`）。
- **无评判检索运行器**（`scripts/run_s1_retrieval_ab.py`）：文档级标准答案使检索精确率/召回率/命中率可以被确定性地评分，**完全不需要 LLM 评判**。每次运行都记录数据集与适配器哈希、确切的流水线定义、软件包版本，以及对每个结果产物逐一哈希的清单。
- **后端接缝**（`src/utils.py`）：所有模型访问都经由 `ollama|hf` 工厂函数，按组件通过环境变量选择，因此同样的流水线可以在无法运行 Ollama 的机器上运行（§5）。
- **测试与 CI：** 适配器、运行器契约和后端接缝均由无模型的单元测试覆盖（`tests/`），CI（`.github/workflows/ci.yml`）在每次推送和拉取请求时运行 lint、测试以及数据和产物验证器——无需任何模型推理。

**适用范围，精确表述：** §1–§2 中的质量指标仅属于已保存的 12 题运行，**不**迁移到该语料。在 11,309 篇文档的规模上，本仓库展示的是数据与评估基础设施——该规模下尚不存在任何检索或答案质量结果。当这些结果产生时，它们将来自受追踪的运行器，附带各自的出处记录，置于各自标注清楚的表格中。

## 4. 架构

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
| Layer 4. Versioned Knowledge Bases (data/ - see DATA.md)                         |
|   Controlled: V1 (8 docs) / V2 (9 docs) + 12-question eval set (authored)        |
|   Scale: 498,725 MS MARCO-derived passages (out of git) + 500 eval queries       |
|   Enterprise: 11,309 EnterpriseRAG-Bench docs (out of git, adapter-regenerated)  |
|               + 130-question answerable pool (tracked)                           |
+----------------------------------------------------------------------------------+
```

回归测试器（`src/regression_tester.py`）让同一条流水线分别对两个知识库版本运行，并将逐题指标差值分桶为改善 / 下降 / 稳定。EnterpriseRAG 适配器（`scripts/adapters/`）为第 4 层供数，因此同样的流水线和评估器可以不加改动地在企业语料上运行。

## 5. 快速开始

### 前置条件

- Python 3.11（固定于 `.python-version`）
- [Ollama](https://ollama.com/)
- 足够的磁盘空闲空间，用于本地模型、Chroma 持久化和基准测试产物

### 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt   # 精确锁定版本
#（requirements.txt 是较小的顶层规格；lockfile 才是可复现的路径）
```

lockfile 的生成方式和冒烟测试记录见 `docs/A2_ENVIRONMENT.md`。

### 拉取本地模型

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama serve   # 如果尚未运行
```

### 模型后端（`ollama` 为默认，`hf` 可选）

在没有任何配置的情况下，一切都使用上述本地 Ollama 模型——行为保持不变。在无法运行 Ollama 的机器上（例如 NVIDIA 驱动早于当前 Ollama 的 CUDA-12 要求的 GPU 主机），可按组件选择 Hugging Face 后端：

```bash
export RAG_MODEL_BACKEND=hf              # 两个组件同时切换，或按组件分别设置：
export RAG_EMBEDDING_BACKEND=hf          #   仅嵌入（纯检索运行）
export RAG_HF_LLM_MODEL=<org/model-id>       # hf LLM 必需
export RAG_HF_EMBEDDING_MODEL=<org/model-id> # hf 嵌入必需
# 可选：RAG_HF_DEVICE=cuda:0  RAG_HF_MAX_NEW_TOKENS=512  RAG_HF_NORMALIZE_EMBEDDINGS=1
```

这里刻意**不提供默认的 HF 模型 id**：请在首次下载前核实模型卡片（许可证、大小/显存占用），然后再导出 id。`hf` 后端需要完整的 lockfile 环境（`langchain-huggingface` 在其中锁定）；CI 从不触及它。

### 数据

小型受控数据集随仓库发布于 `data/`。可随时校验完整性：

```bash
python scripts/verify_data.py
```

预期结果：脚本对照 `data/MANIFEST.json` 重新校验每个存在的数据文件，任何不匹配都会导致非零退出（`DATA.md` §4），因此干净退出即表示受追踪的数据与清单一致。

两个大型语料不在 git 中；请在本地重新生成：

```bash
python scale_up_dataset.py                        # 流式拉取 MS MARCO v2.1（需要网络）
python scripts/adapters/enterpriserag_s1_to_kb.py # 重建 11,309 篇文档的企业知识库；
                                                  # 需先下载并解压 v1.0.0 切片
                                                  #（URL+校验和：evidence/c2-s1-mac-20260712/）
```

### 启动

```bash
streamlit run app.py
```

预期结果：Streamlit 启动并在本地端口提供仪表盘界面。记录在案的 2026-07-10 无头冒烟测试（`docs/A2_ENVIRONMENT.md`）绑定了 `127.0.0.1:8521` 并返回了 `HTTP/1.1 200 OK`。

### 重新验证已保存的证据

```bash
python scripts/verify_a3.py   # 确定性重校验 -> evidence/verified-2026-07/
```

### 运行单元测试（无需模型）

```bash
pip install -r requirements-ci.txt   # 轻量级：不含 torch/chromadb/ollama
pytest                               # 无模型逻辑；所有 LLM 调用均已被模拟
ruff check .                         # lint
python scripts/verify_data.py        # 对照 data/MANIFEST.json 校验数据完整性
```

测试覆盖无模型的核心逻辑——分块、混合 BM25/稠密检索的合并-去重-重排序、检索指标、回归差异分析、EnterpriseRAG 适配器、检索运行器的出处契约以及后端选择——向量存储、交叉编码器和评判 LLM 均以脚本化替身代替（`tests/dependency_stubs.py` 仅在完整环境缺失时顶替重量级导入）。GitHub Actions（`.github/workflows/ci.yml`）在推送到 `main` 及拉取请求时运行同样的四条命令，外加 `verify_a3.py` 的确定性部分；CI 从不执行模型推理。

## 6. 证据规则与数据出处

上文每个结果表格都带有各自的标注；标注背后的规则如下：

- **确定性事实**（数据集计数、文件校验和、索引/延迟计时、已保存文件的解析）已于 2026-07 在本副本上通过 `scripts/verify_a3.py` 和 `scripts/verify_data.py` 重新验证；输出位于 `evidence/verified-2026-07/`。
- **LLM 评判的质量指标**（§1–§2）产生于已保存的 2026-04 运行，使用通过 Ollama 的本地 `gemma4:e4b` 评判模型，**2026-07 做了重新解析——而非重新评判**。在现有硬件上，用同一评判模型精确重跑已被确认为不可行并就此关闭：笔记本上的评判慢到不具实用性，而 GPU 工作站的 NVIDIA 470 驱动早于当前 Ollama 的 CUDA-12 要求（运行记录：`evidence/workstation-c0c1-20260711/`）。计划中的替代方案是在 Hugging Face 运行时上运行本地评判模型，经由与 §5 相同的后端接缝——它将检验这些*发现*（B 优于 A；V2 更新使 B 退化）在独立评判运行时下能否复现，结果会放在各自标注清楚的表格中报告，绝不混入 2026-04 的列。在此之前，请将来自小型本地评判模型的绝对分数读作流水线/版本之间的相对信号，而非经过校准的绝对值。

数据集出处，每行一条（完整记录：[`DATA.md`](DATA.md)，完整性数值：`data/MANIFEST.json`）：

- 小型知识库和 12 题评估集是**在 `prepare_data.py` 中原创撰写的文本**。
- 191 MB / 498,725 条记录的语料和 500 题评估集由 `scale_up_dataset.py` **衍生自 Microsoft MS MARCO v2.1**，并**保持在 git 之外**：MS MARCO 的条款（2026-07-11 实时核实——“仅限非商业研究目的……不延伸任何许可或其他知识产权”）不授予再分发权利。以清晰标注的合成 schema 样本代替（`data/sample_*_synthetic.json`）；确切引文和发布规则见 `DATA.md` §3。
- EnterpriseRAG-Bench S1 语料和问题池（§3）为 **MIT 许可且完全合成**；适配后的 130 题集受 git 追踪，约 88 MB 的适配知识库可从经哈希校验的切片确定性地重新生成。

## 7. 使用指南

**标签页 1 — 交互式查询（Interactive Query）：** 让一个问题走过流水线 A 或 B；在生成的答案旁查看检索到的文档（id、标题、分数）。

**标签页 2 — 流水线 A/B 对比（Pipeline A/B Comparison）：** 在同一题集上评估两条流水线（数量滑块 2–12）；输出雷达图、分组柱状图、逐题分数表和总结性结论。产物保存于 `results/dashboard_comparison/`。

**标签页 3 — 回归测试（Regression Test）：** 比较同一条流水线在 `knowledge_base_v1.json` 与 `knowledge_base_v2.json` 上的表现。展示指标差值表、改善/下降/稳定计数、逐题状态卡片（V1 与 V2 的答案、检索到的文档 id、逐指标差值）以及知识库变更摘要（新增/修改/删除/未变的文档）。阈值：任一指标 < −0.1 记为下降，任一指标 > +0.1 记为改善，否则为稳定。产物保存于 `results/dashboard_regression/`。

**标签页 4 — 规模与性能（Scale & Performance）：** 对 1K/5K/10K/50K 语料子集的索引基准测试，以及纯检索延迟基准测试（对缓存的 50K 索引执行 100 个确定性查询）。产物保存于 `results/scale_performance/`。

对于企业语料，无评判的检索 A/B 从命令行运行：

```bash
python scripts/run_s1_retrieval_ab.py --help   # 流水线、问题数量上限、输出目录
```

## 8. 使用自定义流水线扩展

流水线共享同一份契约（`src/rag_pipelines.py`）：

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

要添加流水线 C：实现类，在 `app.py` 中添加带缓存的工厂函数，在 `PIPELINE_CLASS_MAP` 和 `PIPELINE_APP_CONFIG` 中注册，并更新流水线选择器。（A/B 对比流程目前硬编码为两条流水线；泛化 `compute_comparison_evaluation()` 及其图表是已知的扩展点。）

`RAGEvaluator.load_eval_dataset()` 要求每个评估条目具有 `question`、`ground_truth` 和 `relevant_doc_ids`（后者可以为空，如重新生成的 `data/large_eval_questions.json`）。数据文件 schema 在 `DATA.md` 和合成样本中有带示例的文档说明。

## 9. 项目结构

```text
rag-quality-lab/
├── app.py                        # Streamlit 仪表盘（四种模式）
├── src/
│   ├── utils.py                  # 模型后端接缝（ollama|hf）、JSON 加载、分块
│   ├── rag_pipelines.py          # BaseRAGPipeline、NaiveVectorRAG、HybridRerankRAG
│   ├── evaluation_engine.py      # RAGEvaluator（RAGAS + 本地评判回退）
│   └── regression_tester.py      # RegressionTester 与报告生成
├── data/
│   ├── knowledge_base_v1.json    # 8 篇文档的基线知识库（原创撰写）
│   ├── knowledge_base_v2.json    # 9 篇文档的更新后知识库（原创撰写）
│   ├── eval_questions.json       # 12 题受控评估集（原创撰写）
│   ├── eval_questions_regression_debug.json  # 4 题调试子集
│   ├── eval_questions_enterpriserag_s1.json  # 130 题企业问题池（MIT，经适配）
│   ├── sample_*_synthetic.json   # 带标注的合成 schema 样本
│   └── MANIFEST.json             # 校验和 / 大小 / 记录数
├── results/                      # 已保存的 2026-04 实验产物（CSV/JSON）
├── evidence/                     # 验证输出 + 获取记录
├── scripts/
│   ├── verify_a3.py              # 对已保存产物的确定性重校验
│   ├── verify_data.py            # 对照 MANIFEST.json 校验数据完整性
│   ├── run_s1_retrieval_ab.py    # 无评判的企业检索 A/B（确定性）
│   ├── adapters/                 # EnterpriseRAG-Bench -> 实验室 schema 转换器
│   └── ci/run_verify_a3_deterministic.py  # CI 包装器（顶替重量级依赖）
├── tests/                        # 无模型单元测试（LLM 调用已模拟）
├── tools/                        # 仪表盘资产导出辅助脚本
├── .github/workflows/ci.yml      # lint + 测试 + 验证器；无模型推理
├── docs/                         # A1 副本说明，A2 环境说明
├── DATA.md                       # 数据集出处、许可、验证协议
├── prepare_data.py               # 构建（其本身即是）小型数据集的源文本
├── scale_up_dataset.py           # 流式拉取 MS MARCO v2.1 -> 大语料 + 评估集
├── pyproject.toml                # ruff + pytest 配置
├── requirements.txt              # 顶层依赖规格
├── requirements-ci.txt           # 轻量级 CI/测试依赖
└── requirements-lock-py311.txt   # 锁定的可复现 lockfile
```

（完整语料、Chroma 存储、虚拟环境和生成的运行时产物均按设计被 gitignore——见 `.gitignore` 和 `docs/A1_COPY_NOTES.md`。）

## 10. 技术栈

| 组件 | 技术 | 版本（lockfile） | 用途 |
| --- | --- | --- | --- |
| 语言 | Python | 3.11 | 核心实现语言 |
| 仪表盘 | Streamlit | 1.45.1 | 交互式本地 UI |
| RAG 框架 | LangChain | 0.3.25 | 提示词、文档、编排原语 |
| LangChain 集成 | `langchain-community` | 0.3.24 | 社区集成 |
| Ollama 集成 | `langchain-ollama` | 0.3.3 | 本地 LLM 与嵌入访问 |
| 向量存储集成 | `langchain-chroma` | 0.2.6 | Chroma 的 LangChain 封装 |
| 向量数据库 | ChromaDB | 1.5.7 | 持久化向量检索 |
| 评估框架 | RAGAS | 0.2.15 | 自动化 RAG 评估 |
| 数据集加载器 | `datasets` | 3.6.0 | 流式 MS MARCO 摄取 |
| 词法检索 | `rank-bm25` | 0.2.2 | BM25 候选检索 |
| 重排序器 | `sentence-transformers` | 4.1.0 | 交叉编码器重排序 |
| 可视化 | Plotly | 6.1.2 | 图表与基准测试图形 |
| 数据分析 | Pandas | 2.2.3 | 表格结果处理 |
| 数值工具 | NumPy | 1.26.4 | 数值支持 |
| 另有锁定 | `langchain-huggingface`、`openpyxl` | 0.2.0、3.1.5 | HF 嵌入封装、Excel 导出 |

默认模型配置（`src/utils.py`，`ollama` 后端）：LLM `gemma4:e4b`，嵌入 `nomic-embed-text`，Ollama 端点 `http://127.0.0.1:11434`；`hf` 后端（§5）将这些替换为环境变量指定的 Hugging Face 模型。流水线 B 在首次运行时还会额外下载 `cross-encoder/ms-marco-MiniLM-L-6-v2`。

## 11. 局限性（如实说明适用范围）

1. **评判模型的保真度与时效性。** 质量指标来自已保存的 2026-04 运行中的小型本地评判模型（通过 Ollama 的 `gemma4:e4b`），2026-07 做了重新解析但未重新评判；用同一评判模型精确重跑在现有硬件上已被确认为不可行并就此关闭（§6）。在计划中的 HF 运行时评判通道运行之前，请谨慎对待绝对分数——流水线之间和知识库版本之间的差值才是有意义的信号。
2. **受控评估的规模。** 高保真的 A/B 与回归工作流运行在一个小型手工撰写的 12 题集上；500 题集驱动的是规模基准测试，而非评判式质量运行。
3. **企业规模下尚无质量结果。** 11,309 篇文档的范围（§3）是经证据验证的数据与评估基础设施；其第一批检索数字将由 `scripts/run_s1_retrieval_ab.py` 产出，并附带各自的出处记录。
4. **单机范围。** 一切都在一台本地机器上运行；已测量的规模故事上限是笔记本上的 50K 文档索引。
5. **受许可限制的数据发布。** MS MARCO 经核实的条款（仅限非商业研究、不授予再分发权利——`DATA.md` §3）意味着 MS MARCO 衍生内容不会进入任何公开发布；仓库改用合成样本来记录 schema。
6. **尚无 UI 截图。** 仪表盘截图将从实际运行的会话中截取并添加；在此之前，不做任何重构或伪造的截图。

## 12. 参考文献

- *Corrective Retrieval Augmented Generation*. arXiv, 2024. <https://arxiv.org/abs/2401.15884>
- *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv, 2023. <https://arxiv.org/abs/2309.15217>
- *Okapi at TREC-3*. NIST TREC-3 Proceedings, 1994. <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>
- *Passage Re-ranking with BERT*. arXiv, 2019. <https://arxiv.org/abs/1901.04085>
- *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv, 2020. <https://arxiv.org/abs/2002.10957>
- *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv, 2016. <https://arxiv.org/abs/1611.09268>

## 13. 权利声明

本仓库当前未授予任何开源许可证；保留所有权利。数据集许可证单独且严格地受其各自条款约束：本仓库受约束的 MS MARCO 非商业、禁止再分发条款，以及受追踪的企业级数据与评估基础设施所使用的 EnterpriseRAG-Bench 切片的 MIT 许可证，均见 `DATA.md`。
