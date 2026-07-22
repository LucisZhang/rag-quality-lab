[English](README.md) | [简体中文](README.zh-CN.md)

# 🔬 RAG 质量实验室（RAG Quality Lab）

一个完全本地化的评估平台，用于度量、对比和回归测试检索增强生成（RAG）流水线——旨在回答团队常常跳过的那个问题：**那次改动真的让答案质量变好了，还是悄悄地让它变差了？**

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5A3E85?style=flat-square)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-0F766E?style=flat-square)
![Ollama Gemma 4](https://img.shields.io/badge/Ollama%2FGemma%204-Local%20LLM-111827?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)

## 0. 当前状态一览（2026-07）

本项目中并存两条工作线：

- **公共 `main`（即本 README）：** 受控/MS MARCO 时代的评估平台——12 题 A/B 与回归工作流、2026-04 保存的运行结果，以及 `evidence/verified-2026-07/` 下 2026-07 的确定性再验证。
- **未同步的本地 C2 检查点（尚未进入本仓库）：** 一个更晚的证据检查点，将实验室重建于 EnterpriseRAG-Bench v1.0.0 S1 范围之上（合成的 Confluence 与 Jira 数据源）——11,309 篇文档（5,189 Confluence + 6,120 Jira）、一个可确定性重建且不纳入 Git 的语料库（由经哈希校验、MIT 许可的源切片重建）、130 道带文档级标准答案（ground truth）的可回答问题、一个后端感知的检索契约、68 个通过的无模型测试，以及一个无评审器（judge-free）的检索运行器契约。C2 是数据与基础设施证据底线：它验证数据集适配与评估管线，**而非**检索或答案质量。在该范围上的 C3 A/B 时间盒刻意以一条**明确的“无结果”记录**收尾，而非来自替代流水线的指标——不应从中推断任何检索、答案质量、评审或回退（fallback）结果。本项目的核心规则：没有指标也好过用错误的技术栈产出指标。

在 C2 同步被审阅并完成之前，公共 `main` 是基线：下文任何内容都不应被解读为声称 S1 语料、S1 测试或任何 S1 质量结果已在 `main` 上线。§3 中的历史 12 题对比**不**迁移到 S1 范围。

**按读者类型的入口：** 评估证据 → 先读 §2，再读 [`evidence/verified-2026-07/`](evidence/verified-2026-07/README.md) 与 [`DATA.md`](DATA.md)；运行实验室 → 先读 §7 快速开始，再读 §8 使用指南；阅读或扩展代码 → §5 架构、§9 自定义流水线，再看 `src/` 与 `tests/`。本仓库是一个运行中的单机评估实验室，而非已发布或带版本的软件包；证据新鲜度在 §2 中按声明逐条跟踪，下一个排队中的证据事项是在工作站上进行全新的评审复跑（Track C0）。

## 1. 核心发现：一次“无害”的知识库更新降低了质量，而实验室抓到了它

一次知识库更新（V1 → V2：修订两篇文档、新增一篇——正是生产 RAG 系统不断吸收的常规内容刷新）**使最强流水线在每一项质量指标上全面退化**，其中忠实度（faithfulness）退化最剧烈。代码没有任何改动，只有文档变了。

| 指标（流水线 B，12 题集） | V1（基线） | V2（更新后） | 差值 |
| --- | ---: | ---: | ---: |
| 忠实度（Faithfulness） | 0.988 | 0.867 | **−0.121** |
| 答案相关性（Answer Relevancy） | 0.973 | 0.888 | −0.086 |
| 上下文精确率（Context Precision） | 0.913 | 0.850 | −0.063 |
| 上下文召回率（Context Recall） | 0.925 | 0.842 | −0.083 |
| 答案正确性（Answer Correctness） | 0.921 | 0.846 | −0.075 |

逐题分桶：**4 题退化 / 0 题改善 / 8 题稳定**（任一指标下降超过 0.1 即记为*退化*）。没有回归工具链，这种退化会不知不觉地上线给用户；有了它，这只是一个可审阅的 diff（`results/dashboard_regression/`）。

*证据标签：* LLM 评审分数来自 **2026-04 保存的运行**（本地 `gemma4:e4b` 评审器），并于 **2026-07 确定性重新解析**（`scripts/verify_a3.py` → `evidence/verified-2026-07/`）；**全新评审再验证待进行（工作站）**。4/0/8 分桶与所有计数均已确定性再验证。

## 2. 证据状态——引用任何数字前请先读本节

本仓库区分两类声明，下文每张结果表都带有其标签：

- **确定性事实**（数据集计数、文件校验和、索引/延迟计时、已保存文件解析）：已于 2026-07 在本副本上通过 `scripts/verify_a3.py` 与 `scripts/verify_data.py` 再验证；输出在 `evidence/verified-2026-07/`。
- **LLM 评审质量指标**（忠实度、相关性、精确率、召回率、正确性）：产生于 **2026-04 保存的运行**，使用本地 `gemma4:e4b` 评审器，并在 2026-07 **重新解析——而非重新评审**。全新的评审复跑已排为首个工作站接入任务（Track C0）。来自小型本地评审器的绝对分数应被视为流水线/版本之间的相对信号，而非校准过的绝对值。

数据集来源、许可状态与完整性校验和见 [`DATA.md`](DATA.md) 与 `data/MANIFEST.json`。

## 3. 流水线 A/B 对比——质量与延迟，两侧都有度量

同一 12 题评估集、同一知识库、同一评审器；仅检索架构不同：

- **流水线 A —— NaiveVectorRAG：** Chroma 相似度搜索 → top-k 上下文 → Gemma 4。
- **流水线 B —— HybridRerankRAG：** Chroma top-10 + BM25 top-10 → 合并/去重 → 交叉编码器重排（`cross-encoder/ms-marco-MiniLM-L-6-v2`）→ Gemma 4。

| 指标 | 流水线 A（朴素向量） | 流水线 B（混合+重排） | 差值 |
| --- | ---: | ---: | ---: |
| 忠实度 | 0.888 | 0.988 | +0.100 |
| 答案相关性 | 0.804 | 0.973 | +0.169 |
| 上下文精确率 | 0.784 | 0.913 | +0.129 |
| 上下文召回率 | 0.800 | 0.925 | +0.125 |
| 答案正确性 | 0.771 | 0.921 | +0.150 |
| **总体均值** | **0.809** | **0.944** | **+0.135（相对 +16.6%）** |

*本表全部五项指标仅来自 12 题受控集——请将 +16.6% 的提升视为小样本集上的架构信号，而非一般性结论。*

流水线 B 在五项指标上全胜；其在该集上的检索诊断召回率/命中率为 1.0。代价见下文 §4：在 5 万文档规模下检索延迟约为流水线 A 的 4.6 倍。这一对数字——质量提升及其延迟成本——正是本实验室要使之可度量的架构决策。

*证据标签：* 评审指标 = 2026-04 保存的运行，2026-07 确定性重新解析（均值 0.8093 → 0.9438，相对提升 +16.6%）；全新评审再验证待进行（工作站）。源产物：`results/dashboard_comparison/`。

## 4. 规模基准（确定性计时，2026-04 保存的运行）

MS MARCO 派生语料上的索引吞吐（嵌入：`nomic-embed-text`，经 Ollama，本地机器）：

| 文档数 | 已索引向量数 | 时间（秒） | 每千文档秒数 |
| ---: | ---: | ---: | ---: |
| 1,000 | 1,121 | 13.22 | 13.22 |
| 5,000 | 5,614 | 60.99 | 12.20 |
| 10,000 | 11,290 | 134.85 | 13.49 |
| 50,000 | 56,039 | 691.17 | 13.82 |

在缓存的 5 万文档索引上的检索延迟（100 条确定性查询）：

| 流水线 | P50（毫秒） | P95（毫秒） | P99（毫秒） | 均值（毫秒） |
| --- | ---: | ---: | ---: | ---: |
| 流水线 A | 28.13 | 37.39 | 39.65 | 29.15 |
| 流水线 B | 128.44 | 188.41 | 212.43 | 134.15 |

*证据标签：* 计时为确定性的已保存产物（`results/scale_performance/*.csv`，2026-04），2026-07 重新解析。它们刻画的是笔记本级机器；按每千文档约 13.8 秒计，对全部 498,725 条记录的语料做嵌入外推约需 1.9 小时——这正是 GPU 工作站运行的量化论据（外推，而非实测）。

## 5. 架构

```text
+----------------------------------------------------------------------------------+
| 第 1 层. Streamlit 仪表盘 (app.py)                                               |
|   交互式查询 | 流水线 A/B 对比 | 回归测试 | 规模与性能                            |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| 第 2 层. 评估引擎 (src/evaluation_engine.py)                                     |
|   RAGEvaluator - 五项 RAGAS 质量指标（忠实度、答案相关性、                        |
|   上下文精确率、上下文召回率、答案正确性）                                        |
|   + 检索诊断（retrieval_precision / recall / hit）                                |
|   后端模式: auto | ragas | fallback（本地 LLM 评审器）                            |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| 第 3 层. RAG 流水线 (src/rag_pipelines.py)                                       |
|   A: NaiveVectorRAG   - Chroma 相似度 -> top-k -> Gemma 4                        |
|   B: HybridRerankRAG  - Chroma top-10 + BM25 top-10 -> 合并/去重                 |
|                         -> 交叉编码器重排 -> Gemma 4                              |
+------------------------------------------+---------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------+
| 第 4 层. 版本化知识库 (data/ - 见 DATA.md)                                       |
|   受控集: V1（8 篇）/ V2（9 篇）+ 12 题评估集（手写）                            |
|   规模集: 498,725 条 MS MARCO 派生段落（不入 git）+ 500 条评估查询               |
+----------------------------------------------------------------------------------+
```

回归工具链（`src/regression_tester.py`）让同一条流水线分别对两个知识库版本运行，并将逐题指标差值分桶为改善 / 退化 / 稳定。

## 6. 数据与许可（摘要——完整记录见 `DATA.md`）

- 小型知识库与 12 题评估集是 **`prepare_data.py` 中原创撰写的文本**。
- 191 MB / 498,725 条记录的语料与 500 题评估集由 `scale_up_dataset.py` **从 Microsoft MS MARCO v2.1 派生**。两者在公共仓库中均**不纳入 git**；完整性通过 `data/MANIFEST.json` 期望计数 + `scripts/verify_data.py` 传递。
- **MS MARCO 条款已于 2026-07-11 实时核验**：“仅限非商业研究目的……不授予任何许可或其他知识产权”（`microsoft/msmarco` 的 `Notice.md`，经 GitHub API 读取）。此处未再发布任何语料记录；以明确标注的合成模式样例代替（`data/sample_*_synthetic.json`）。确切引文与本仓库须遵守的发布规则见 `DATA.md` §3。

## 7. 快速开始

### 前置条件

- Python 3.11（锁定于 `.python-version`）
- [Ollama](https://ollama.com/)
- 足够的空闲磁盘，用于本地模型、Chroma 持久化与基准产物

### 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock-py311.txt   # 精确锁定版本
# （requirements.txt 是更小的顶层规格；lockfile 是可复现路径）
```

lockfile 的生成方式与冒烟测试记录见 `docs/A2_ENVIRONMENT.md`。

### 拉取本地模型

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
ollama serve   # 如尚未运行
```

### 数据

小型受控数据集随 `data/` 附带。可随时验证完整性：

```bash
python scripts/verify_data.py
```

预期结果：脚本将每个存在的数据文件与 `data/MANIFEST.json` 重新核对，任一不匹配即以非零码退出（`DATA.md` §4），因此干净退出即表示被跟踪的数据与清单一致。

大型语料不在 git 中；可重新生成（需要网络，流式读取 MS MARCO v2.1）：

```bash
python scale_up_dataset.py
```

### 启动

```bash
streamlit run app.py
```

预期结果：Streamlit 启动并在本地端口提供仪表盘外壳。2026-07-10 记录的无头冒烟测试（`docs/A2_ENVIRONMENT.md`）绑定 `127.0.0.1:8521` 并返回 `HTTP/1.1 200 OK`。

### 再验证已保存的证据

```bash
python scripts/verify_a3.py   # 确定性再检查 -> evidence/verified-2026-07/
```

### 运行单元测试（无需模型）

```bash
pip install -r requirements-ci.txt   # 轻量：不含 torch/chromadb/ollama
pytest                               # 无模型逻辑；所有 LLM 调用均打桩
ruff check .                         # 代码检查
python scripts/verify_data.py        # 数据完整性 vs data/MANIFEST.json
```

测试覆盖无模型核心——分块、混合 BM25/稠密向量的合并-去重-重排、检索指标与回归 diff——以脚本化替身取代向量库、交叉编码器与评审 LLM（`tests/dependency_stubs.py` 仅在完整环境缺失时代理重型导入）。GitHub Actions（`.github/workflows/ci.yml`）在向 `main` 推送及拉取请求时运行同样四条命令外加 `verify_a3.py` 的确定性部分；CI 从不执行模型推理。

## 8. 使用指南

**标签页 1 —— 交互式查询：** 用流水线 A 或 B 运行一个问题；在生成答案旁查看检索到的文档（id、标题、分数）。

**标签页 2 —— 流水线 A/B 对比：** 在同一题集上评估两条流水线（数量滑块 2–12）；输出雷达图、分组柱状图、逐题分数表与总结结论。产物落在 `results/dashboard_comparison/`。

**标签页 3 —— 回归测试：** 让同一条流水线跨 `knowledge_base_v1.json` 与 `knowledge_base_v2.json` 对比。展示指标差值表、改善/退化/稳定计数、逐题状态卡片（V1 vs V2 答案、检索到的文档 id、逐项指标差值）以及知识库变更摘要（新增/修改/删除/未变文档）。阈值：任一指标 < −0.1 为退化，任一指标 > +0.1 为改善，否则为稳定。产物落在 `results/dashboard_regression/`。

**标签页 4 —— 规模与性能：** 在 1K/5K/10K/50K 语料子集上的索引基准，以及检索-only 延迟基准（对缓存的 5 万文档索引运行 100 条确定性查询）。产物落在 `results/scale_performance/`。

## 9. 以自定义流水线扩展

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

要添加流水线 C：实现该类、在 `app.py` 中添加带缓存的工厂、将其注册到 `PIPELINE_CLASS_MAP` 与 `PIPELINE_APP_CONFIG`，并更新流水线选择器。（A/B 对比流程目前硬编码为两条流水线；泛化 `compute_comparison_evaluation()` 及其图表是已知的扩展点。）

`RAGEvaluator.load_eval_dataset()` 要求每条评估项包含 `question`、`ground_truth` 与 `relevant_doc_ids`（后者可为空，如重新生成的 `data/large_eval_questions.json`）。数据文件模式在 `DATA.md` 与合成样例中有带示例的文档说明。

## 10. 项目结构

```text
rag-quality-lab/
├── app.py                        # Streamlit 仪表盘（四种模式）
├── src/
│   ├── utils.py                  # Ollama 客户端、JSON 加载、分块、Chroma 创建
│   ├── rag_pipelines.py          # BaseRAGPipeline、NaiveVectorRAG、HybridRerankRAG
│   ├── evaluation_engine.py      # RAGEvaluator（RAGAS + 本地评审回退）
│   └── regression_tester.py      # RegressionTester 与报告生成
├── data/
│   ├── knowledge_base_v1.json    # 8 篇文档的基线知识库（手写）
│   ├── knowledge_base_v2.json    # 9 篇文档的更新知识库（手写）
│   ├── eval_questions.json       # 12 题受控评估集（手写）
│   ├── eval_questions_regression_debug.json  # 4 题调试子集
│   ├── sample_*_synthetic.json   # 带标注的合成模式样例
│   └── MANIFEST.json             # 校验和 / 大小 / 记录数
├── results/                      # 2026-04 保存的实验产物（CSV/JSON）
├── evidence/verified-2026-07/    # 确定性再验证输出 + 尝试记录
├── scripts/
│   ├── verify_a3.py              # 对已保存产物的确定性再检查
│   ├── verify_data.py            # 数据完整性 vs MANIFEST.json
│   └── ci/run_verify_a3_deterministic.py  # CI 包装器（打桩重依赖）
├── tests/                        # 无模型单元测试（LLM 调用打桩）
├── tools/                        # 仪表盘资产导出辅助脚本
├── .github/workflows/ci.yml      # lint + 测试 + 校验器；无模型推理
├── docs/                         # A1 文案说明、A2 环境说明
├── DATA.md                       # 数据集来源、许可、验证规程
├── prepare_data.py               # 构建（且其本身就是）小型数据集的源文本
├── scale_up_dataset.py           # 流式读取 MS MARCO v2.1 -> 大型语料 + 评估集
├── pyproject.toml                # ruff + pytest 配置
├── requirements.txt              # 顶层依赖规格
├── requirements-ci.txt           # 轻量 CI/测试依赖
└── requirements-lock-py311.txt   # 锁定的可复现 lockfile
```

（完整语料、Chroma 存储、虚拟环境与运行时产物按设计被 gitignore——见 `.gitignore` 与 `docs/A1_COPY_NOTES.md`。）

## 11. 技术栈

| 组件 | 技术 | 版本（lockfile） | 用途 |
| --- | --- | --- | --- |
| 语言 | Python | 3.11 | 核心实现语言 |
| 仪表盘 | Streamlit | 1.45.1 | 交互式本地 UI |
| RAG 框架 | LangChain | 0.3.25 | 提示、文档、编排原语 |
| LangChain 集成 | `langchain-community` | 0.3.24 | 社区集成 |
| Ollama 集成 | `langchain-ollama` | 0.3.3 | 本地 LLM 与嵌入访问 |
| 向量库集成 | `langchain-chroma` | 0.2.6 | Chroma 的 LangChain 封装 |
| 向量数据库 | ChromaDB | 1.5.7 | 持久化向量检索 |
| 评估框架 | RAGAS | 0.2.15 | 自动化 RAG 评估 |
| 数据集加载器 | `datasets` | 3.6.0 | 流式接入 MS MARCO |
| 词法检索 | `rank-bm25` | 0.2.2 | BM25 候选检索 |
| 重排器 | `sentence-transformers` | 4.1.0 | 交叉编码器重排 |
| 可视化 | Plotly | 6.1.2 | 图表与基准绘图 |
| 数据分析 | Pandas | 2.2.3 | 表格结果处理 |
| 数值工具 | NumPy | 1.26.4 | 数值支持 |
| 另已锁定 | `langchain-huggingface`、`openpyxl` | 0.2.0、3.1.5 | HF 嵌入封装、Excel 导出 |

本地模型配置（`src/utils.py`）：LLM `gemma4:e4b`，嵌入 `nomic-embed-text`，Ollama 端点 `http://127.0.0.1:11434`。流水线 B 在首次运行时还会下载 `cross-encoder/ms-marco-MiniLM-L-6-v2`。

## 12. 局限（诚实范围）

1. **评审器保真度与新鲜度。** 质量指标来自 2026-04 保存运行中的小型本地评审器（`gemma4:e4b`）。它们已于 2026-07 确定性重新解析，但**尚未重新评审**；全新的评审复跑已排队等待 GPU 工作站（2026-07 的一次 1 题本地复评尝试确认笔记本路径慢得不可行，该尝试已被关闭而非强行推进）。请谨慎看待绝对分数；流水线之间与知识库版本之间的差值才是有意义的信号。
2. **受控评估规模。** 高保真 A/B 与回归工作流运行在手写的小型 12 题集上；500 题集用于驱动规模基准，而非评审质量运行。
3. **单机范围。** 一切运行在一台本地机器上；规模故事的上限是笔记本上的 5 万文档索引。更大规模的运行是计划中的工作站轨道，有其自身的证据纪律。
4. **许可限制的数据发布。** 经核验的 MS MARCO 条款（仅限非商业研究、无再分发权——`DATA.md` §3）意味着 MS MARCO 派生内容不得进入任何公开发布；本仓库改用合成样例来记录模式。
5. **尚无 UI 截图。** 仪表盘截图将从实时会话中截取并添加；在此之前不重建也不摆拍任何截图。

## 13. 参考文献

- *Corrective Retrieval Augmented Generation*. arXiv, 2024. <https://arxiv.org/abs/2401.15884>
- *Ragas: Automated Evaluation of Retrieval Augmented Generation*. arXiv, 2023. <https://arxiv.org/abs/2309.15217>
- *Okapi at TREC-3*. NIST TREC-3 Proceedings, 1994. <https://trec.nist.gov/pubs/trec3/t3_proceedings.html>
- *Passage Re-ranking with BERT*. arXiv, 2019. <https://arxiv.org/abs/1901.04085>
- *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv, 2020. <https://arxiv.org/abs/2002.10957>
- *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. arXiv, 2016. <https://arxiv.org/abs/1611.09268>

## 14. 权利声明

本仓库目前未授予任何开源许可；保留所有权利。数据集许可单独且严格地受其自身条款约束：本仓库须遵守的 MS MARCO 非商业、禁止再分发条款，以及未同步本地 C2 检查点所用 EnterpriseRAG-Bench 切片的 MIT 许可，均见 `DATA.md`。
