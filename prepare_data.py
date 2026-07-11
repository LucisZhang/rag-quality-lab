"""
准备 RAG 质量实验室的知识库和评测数据集。
使用 Wikipedia 文章作为知识库，手动构建评测 QA 对。
"""
import json
import os

os.makedirs("data", exist_ok=True)

# ===== 知识库文档（V1 版本）=====
# 使用若干个知识领域的短文作为知识库
knowledge_base_v1 = [
    {
        "id": "doc_001",
        "title": "Transformer Architecture",
        "content": """The Transformer is a deep learning architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. Unlike recurrent neural networks (RNNs), Transformers process all positions in a sequence simultaneously using a mechanism called self-attention. The architecture consists of an encoder and a decoder, each made up of multiple layers. Each layer in the encoder has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. The decoder has an additional cross-attention sub-layer. Transformers use positional encoding to inject information about the position of tokens in the sequence, since the architecture itself has no inherent notion of order. The self-attention mechanism computes attention scores between all pairs of positions, allowing the model to capture long-range dependencies efficiently. Multi-head attention runs multiple attention functions in parallel, allowing the model to attend to information from different representation subspaces. The Transformer has become the foundation for most modern NLP models, including BERT, GPT, and T5."""
    },
    {
        "id": "doc_002",
        "title": "Retrieval-Augmented Generation",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by combining them with external knowledge retrieval. First introduced by Lewis et al. in 2020, RAG addresses the limitation of LLMs having static, potentially outdated knowledge. In a RAG system, when a query is received, a retriever component searches a knowledge base to find relevant documents. These retrieved documents are then provided as context to the language model, which generates an answer grounded in the retrieved information. The retrieval step typically uses dense vector representations (embeddings) of both the query and documents, computing similarity scores to find the most relevant passages. RAG systems can be categorized into naive RAG (simple retrieve-then-generate), advanced RAG (with query rewriting, re-ranking, and filtering), and modular RAG (with interchangeable components). Key challenges in RAG include retrieval quality, context window limitations, faithfulness of generated answers to retrieved context, and handling conflicting information across sources."""
    },
    {
        "id": "doc_003",
        "title": "Vector Databases",
        "content": """Vector databases are specialized database systems designed to store, index, and query high-dimensional vector data efficiently. They are essential infrastructure for AI applications that rely on similarity search, such as recommendation systems, image retrieval, and RAG systems. Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and ChromaDB. These databases use approximate nearest neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), and PQ (Product Quantization) to enable fast similarity search at scale. Unlike traditional databases that use exact matching, vector databases find items that are semantically similar by computing distances (cosine similarity, Euclidean distance, or dot product) between vector representations. Key features include support for metadata filtering, hybrid search (combining vector and keyword search), real-time indexing, and horizontal scaling. ChromaDB is an open-source embedding database that emphasizes simplicity and developer experience, supporting both in-memory and persistent storage modes."""
    },
    {
        "id": "doc_004",
        "title": "Evaluation Metrics for RAG Systems",
        "content": """Evaluating RAG systems requires measuring both retrieval quality and generation quality. Common retrieval metrics include context precision (what fraction of retrieved documents are relevant), context recall (what fraction of relevant documents are retrieved), and Mean Reciprocal Rank (MRR). For generation quality, key metrics include faithfulness (whether the generated answer is supported by the retrieved context), answer relevancy (whether the answer addresses the question), and answer correctness (whether the answer matches the ground truth). The RAGAS framework provides automated evaluation using LLM-as-judge approaches, where a language model assesses these qualities. Beyond automated metrics, human evaluation remains important for assessing aspects like fluency, completeness, and usefulness. A/B testing in production environments can measure real-world impact metrics such as user satisfaction and task completion rates. Regression testing is crucial for monitoring quality over time, especially when updating knowledge bases or changing retrieval strategies."""
    },
    {
        "id": "doc_005",
        "title": "BM25 and Hybrid Search",
        "content": """BM25 (Best Matching 25) is a probabilistic ranking function used in information retrieval. It is an improvement over the classic TF-IDF approach, incorporating document length normalization and term frequency saturation. BM25 scores documents based on the query terms appearing in each document, considering term frequency, inverse document frequency, and document length. Despite the rise of neural retrievers, BM25 remains competitive for many search tasks, especially for keyword-heavy queries and exact term matching. Hybrid search combines BM25 (lexical search) with dense vector retrieval (semantic search) to leverage the strengths of both approaches. In hybrid search, scores from both methods are typically combined using reciprocal rank fusion (RRF) or linear interpolation. Cross-encoder re-ranking is often applied as a second stage: after retrieving candidates from both BM25 and vector search, a cross-encoder model scores each query-document pair more carefully to produce a final ranking. This two-stage approach (retrieve then re-rank) achieves better results than either method alone while remaining computationally feasible."""
    },
    {
        "id": "doc_006",
        "title": "Large Language Models Overview",
        "content": """Large Language Models (LLMs) are neural networks trained on massive text corpora to understand and generate human language. Modern LLMs are based on the Transformer architecture and are typically trained using next-token prediction (autoregressive models like GPT) or masked language modeling (models like BERT). Key milestones include GPT-3 (175 billion parameters, 2020), ChatGPT (RLHF-trained GPT-3.5, 2022), GPT-4 (multimodal, 2023), and open-source alternatives like LLaMA, Mistral, and Qwen. LLMs exhibit emergent capabilities such as in-context learning, chain-of-thought reasoning, and instruction following. Fine-tuning techniques include supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and parameter-efficient methods like LoRA. Challenges include hallucination (generating plausible but incorrect information), context window limitations, computational costs, and safety concerns. The trend toward smaller, more efficient models (like Phi, Gemma) shows that capability can be achieved without extreme scale."""
    },
    {
        "id": "doc_007",
        "title": "Knowledge Graphs and Structured Retrieval",
        "content": """Knowledge graphs represent information as entities (nodes) and relationships (edges), forming a structured network of facts. Major knowledge graphs include Wikidata, Google Knowledge Graph, and domain-specific graphs in medicine, finance, and science. In the context of RAG, knowledge graphs enable structured retrieval that can capture multi-hop relationships that vector similarity search might miss. GraphRAG is an approach that combines graph-based retrieval with language model generation. The process typically involves: extracting entities and relationships from text, constructing or updating a knowledge graph, performing graph queries (e.g., finding paths between entities), and using the retrieved subgraph as context for generation. Advantages of graph-based retrieval include better handling of complex relational queries, explicit reasoning chains, and reduced hallucination through structured evidence. However, challenges include the cost of graph construction, maintaining graph quality, and scaling to large knowledge bases."""
    },
    {
        "id": "doc_008",
        "title": "Chunking Strategies for RAG",
        "content": """Chunking is the process of splitting documents into smaller segments for indexing in a RAG system. The choice of chunking strategy significantly impacts retrieval quality. Common approaches include fixed-size chunking (splitting by character or token count with overlap), sentence-based chunking (splitting at sentence boundaries), paragraph-based chunking (using document structure), and semantic chunking (grouping semantically similar sentences together). Key parameters include chunk size (typically 256-1024 tokens) and chunk overlap (typically 10-20% of chunk size). Smaller chunks provide more precise retrieval but may lack context; larger chunks preserve context but may include irrelevant information. Advanced strategies include hierarchical chunking (maintaining parent-child relationships between chunks), agentic chunking (using an LLM to determine optimal split points), and late chunking (embedding full documents first, then splitting). The optimal strategy depends on the document type, query patterns, and downstream task requirements."""
    }
]

# ===== 知识库文档（V2 版本 — 模拟更新）=====
# 修改部分文档内容，模拟知识库的演进
knowledge_base_v2 = []
for doc in knowledge_base_v1:
    knowledge_base_v2.append(doc.copy())

# 更新 doc_002：添加新信息，修改部分旧信息
knowledge_base_v2[1] = {
    "id": "doc_002",
    "title": "Retrieval-Augmented Generation",
    "content": """Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by combining them with external knowledge retrieval. First introduced by Lewis et al. in 2020, RAG addresses the limitation of LLMs having static, potentially outdated knowledge. In a RAG system, when a query is received, a retriever component searches a knowledge base to find relevant documents. These retrieved documents are then provided as context to the language model, which generates an answer grounded in the retrieved information. Modern RAG systems in 2025-2026 have evolved significantly: agentic RAG uses AI agents to dynamically decide retrieval strategies; corrective RAG (CRAG) adds a verification step to assess retrieval quality; and self-RAG enables the model to decide when retrieval is necessary. Key challenges include retrieval quality, faithfulness hallucination, conflicting information, and the cost-latency tradeoff of multi-stage pipelines. The field is moving from simple retrieve-and-generate toward complex orchestration systems with query routing, adaptive retrieval, and built-in evaluation loops."""
}

# 更新 doc_004：修改评测指标描述
knowledge_base_v2[3] = {
    "id": "doc_004",
    "title": "Evaluation Metrics for RAG Systems",
    "content": """Evaluating RAG systems requires measuring both retrieval quality and generation quality across multiple dimensions. For retrieval, the primary metrics are context precision (fraction of retrieved passages that are relevant), context recall (fraction of ground-truth relevant passages that are retrieved), and NDCG (Normalized Discounted Cumulative Gain). For generation, faithfulness measures whether claims in the answer are supported by retrieved context, answer relevancy measures topical alignment with the question, and answer correctness compares against ground truth references. The RAGAS framework (2024-2026) has become the de facto standard for automated RAG evaluation, using LLM-as-judge methodology. Newer evaluation approaches include reference-free evaluation (no ground truth needed), component-level diagnostics (pinpointing whether failures come from retrieval, context selection, or generation), and adversarial testing (probing for hallucination under edge cases). Production monitoring should track metric drift over time, with automated regression tests triggered on knowledge base updates or pipeline configuration changes."""
}

# 添加一篇新文档
knowledge_base_v2.append({
    "id": "doc_009",
    "title": "Agentic RAG and Multi-Step Retrieval",
    "content": """Agentic RAG represents the evolution of RAG systems from static pipelines to dynamic, agent-driven workflows. In agentic RAG, an AI agent orchestrates the retrieval process by analyzing the query, planning retrieval steps, executing searches across multiple sources, evaluating results, and iterating if needed. Key patterns include: (1) Query decomposition - breaking complex questions into sub-queries; (2) Adaptive retrieval - choosing between vector search, keyword search, or structured queries based on query type; (3) Self-reflection - the agent checks whether retrieved context is sufficient before generating; (4) Tool use - the agent can call APIs, databases, or calculators as needed. Frameworks like LangGraph and CrewAI enable building agentic RAG systems with state management and multi-step reasoning. While agentic RAG improves answer quality for complex queries, it introduces challenges in latency (multiple LLM calls), cost, debugging complexity, and evaluation (non-deterministic agent paths make regression testing harder)."""
})

# ===== 评测问答对 =====
eval_questions = [
    {
        "question": "What is the self-attention mechanism in Transformers and why is it important?",
        "ground_truth": "Self-attention is a mechanism that computes attention scores between all pairs of positions in a sequence, allowing the model to capture long-range dependencies efficiently. Unlike RNNs, it processes all positions simultaneously.",
        "relevant_doc_ids": ["doc_001"]
    },
    {
        "question": "What are the main components of a RAG system?",
        "ground_truth": "A RAG system consists of a retriever component that searches a knowledge base for relevant documents, and a language model that generates answers grounded in the retrieved information. The retrieval typically uses dense vector embeddings to find similar passages.",
        "relevant_doc_ids": ["doc_002"]
    },
    {
        "question": "How does BM25 differ from vector-based retrieval, and why is hybrid search effective?",
        "ground_truth": "BM25 is a probabilistic lexical ranking function based on term frequency and document length, while vector retrieval uses dense embeddings for semantic similarity. Hybrid search combines both to leverage BM25's strength in exact term matching with vector search's semantic understanding. Re-ranking with cross-encoders further improves results.",
        "relevant_doc_ids": ["doc_005"]
    },
    {
        "question": "What is faithfulness in the context of RAG evaluation?",
        "ground_truth": "Faithfulness measures whether the generated answer is supported by the retrieved context. It assesses if the claims in the answer can be traced back to the information in the retrieved documents, rather than being hallucinated.",
        "relevant_doc_ids": ["doc_004"]
    },
    {
        "question": "What are the different chunking strategies for RAG and how do they affect retrieval quality?",
        "ground_truth": "Common chunking strategies include fixed-size, sentence-based, paragraph-based, and semantic chunking. Smaller chunks provide more precise retrieval but may lack context, while larger chunks preserve context but may include irrelevant information. Advanced strategies include hierarchical and agentic chunking.",
        "relevant_doc_ids": ["doc_008"]
    },
    {
        "question": "What algorithms do vector databases use for similarity search?",
        "ground_truth": "Vector databases use approximate nearest neighbor (ANN) algorithms including HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), and PQ (Product Quantization) to enable fast similarity search at scale.",
        "relevant_doc_ids": ["doc_003"]
    },
    {
        "question": "What is GraphRAG and how does it improve over standard vector-based RAG?",
        "ground_truth": "GraphRAG combines graph-based retrieval with language model generation. It extracts entities and relationships into a knowledge graph, enabling structured retrieval that captures multi-hop relationships that vector similarity search might miss. Advantages include better handling of complex relational queries and explicit reasoning chains.",
        "relevant_doc_ids": ["doc_007"]
    },
    {
        "question": "What are the main challenges of large language models?",
        "ground_truth": "Key challenges include hallucination (generating plausible but incorrect information), context window limitations, high computational costs, and safety concerns. These limitations motivate techniques like RAG for grounding outputs in factual sources.",
        "relevant_doc_ids": ["doc_006"]
    },
    {
        "question": "How does RAGAS evaluate RAG systems?",
        "ground_truth": "RAGAS provides automated evaluation using LLM-as-judge approaches, measuring metrics like faithfulness, answer relevancy, context precision, context recall, and answer correctness. It uses a language model to assess these qualities automatically.",
        "relevant_doc_ids": ["doc_004"]
    },
    {
        "question": "What is the difference between naive RAG and advanced RAG?",
        "ground_truth": "Naive RAG uses a simple retrieve-then-generate approach. Advanced RAG adds techniques like query rewriting, re-ranking, and filtering to improve retrieval and generation quality. Modular RAG further introduces interchangeable components for flexible pipeline design.",
        "relevant_doc_ids": ["doc_002"]
    },
    {
        "question": "How does multi-head attention work in Transformers?",
        "ground_truth": "Multi-head attention runs multiple attention functions in parallel, allowing the model to attend to information from different representation subspaces simultaneously. Each head learns different attention patterns.",
        "relevant_doc_ids": ["doc_001"]
    },
    {
        "question": "What is reciprocal rank fusion in hybrid search?",
        "ground_truth": "Reciprocal rank fusion (RRF) is a method for combining scores from multiple retrieval methods (like BM25 and vector search) in hybrid search. It merges the ranked lists from different retrievers into a single ranking.",
        "relevant_doc_ids": ["doc_005"]
    }
]

# 保存
with open("data/knowledge_base_v1.json", "w") as f:
    json.dump(knowledge_base_v1, f, indent=2)

with open("data/knowledge_base_v2.json", "w") as f:
    json.dump(knowledge_base_v2, f, indent=2)

with open("data/eval_questions.json", "w") as f:
    json.dump(eval_questions, f, indent=2)

print("✅ 数据准备完成！")
print(f"   知识库 V1: {len(knowledge_base_v1)} 篇文档")
print(f"   知识库 V2: {len(knowledge_base_v2)} 篇文档 (含 2 篇修改 + 1 篇新增)")
print(f"   评测问题: {len(eval_questions)} 个 QA 对")
