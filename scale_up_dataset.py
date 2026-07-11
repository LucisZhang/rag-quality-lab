"""
扩充评测数据规模，让项目满足大数据体量要求。
使用 Microsoft MS MARCO 数据集（880万段落 + 100万查询的真实 RAG benchmark）。
"""
from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)

# MS MARCO passage ranking - 真实的搜索引擎查询+标注数据
# 完整数据 880 万段落，流式下载避免占满硬盘
ds_corpus = load_dataset(
    "microsoft/ms_marco", "v2.1",
    split="train", streaming=True
)

# 取前 50000 条作为大规模知识库
knowledge_base = []
for i, item in enumerate(ds_corpus):
    if i >= 50000:
        break
    for j, passage in enumerate(item.get("passages", {}).get("passage_text", [])):
        knowledge_base.append({
            "id": f"msmarco_{i}_{j}",
            "title": f"Query {i}",
            "content": passage
        })

# 评测问题 500 条
eval_questions = []
ds_eval = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
for i, item in enumerate(ds_eval):
    if i >= 500: break
    if item.get("answers") and len(item["answers"]) > 0:
        eval_questions.append({
            "question": item["query"],
            "ground_truth": item["answers"][0],
            "relevant_doc_ids": []
        })

with open("data/large_knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f)
with open("data/large_eval_questions.json", "w") as f:
    json.dump(eval_questions, f)

print(f"✅ 大规模数据集就位：")
print(f"   知识库: {len(knowledge_base)} 条段落（约 {len(knowledge_base)*500/1024/1024:.1f} MB）")
print(f"   评测问题: {len(eval_questions)} 个")
