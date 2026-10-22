from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knowledge_rag import KnowledgeRAG
from multi_agent import CoordinatorAgent

rag_out = ROOT / "results" / "knowledge_rag"
agent_out = ROOT / "results" / "multi_agent"
rag_out.mkdir(parents=True, exist_ok=True)
agent_out.mkdir(parents=True, exist_ok=True)

engine = KnowledgeRAG(ROOT / "knowledge")
queries = json.loads((ROOT / "knowledge" / "benchmark.json").read_text(encoding="utf-8"))
methods = ["BM25", "LSA", "KG", "Hybrid", "GraphRAG"]
rows, detail = [], []
for method in methods:
    per_query, latencies = [], []
    for item in queries:
        start = time.perf_counter()
        result = engine.retrieve(item["question"], method=method, top_k=3, hops=2)
        latencies.append((time.perf_counter() - start) * 1000)
        predicted = [d["id"] for d in result["documents"]]
        gold = item["gold"]
        if gold:
            hits = [x for x in predicted if x in gold]
            recall = len(set(hits)) / len(set(gold))
            rank = next((i + 1 for i, x in enumerate(predicted) if x in gold), None)
            mrr = 1 / rank if rank else 0.0
            complete = float(set(gold).issubset(predicted))
            per_query.append((recall, mrr, complete))
        detail.append({"method": method, "id": item["id"], "group": item["group"], "gold": gold, "retrieved": predicted})
    answerable = np.asarray(per_query, dtype=float)
    ood = [d for d in detail if d["method"] == method and d["group"] == "out_of_scope"]
    rows.append({
        "Method": method,
        "Recall@3": float(answerable[:, 0].mean()),
        "MRR@3": float(answerable[:, 1].mean()),
        "Complete@3": float(answerable[:, 2].mean()),
        "OOD_EmptyRate": float(np.mean([len(x["retrieved"]) == 0 for x in ood])),
        "Latency_ms": float(np.median(latencies)),
    })

with (rag_out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
(rag_out / "benchmark.json").write_text(json.dumps({"documents": len(engine.documents), "triples": len(engine.triples), "queries": len(queries), "rows": rows, "detail": detail}, ensure_ascii=False, indent=2), encoding="utf-8")

x = np.arange(len(methods)); width = .25
fig, ax = plt.subplots(figsize=(10, 4.6))
for offset, metric, color in [(-width, "Recall@3", "#c62828"), (0, "MRR@3", "#ef5350"), (width, "Complete@3", "#ff8a80")]:
    ax.bar(x + offset, [r[metric] for r in rows], width, label=metric, color=color)
ax.set_xticks(x, methods); ax.set_ylim(0, 1.08); ax.set_ylabel("Điểm trên 12 câu có nguồn"); ax.legend(ncol=3, loc="lower right"); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(rag_out / "retrieval_comparison.png", dpi=170); plt.close(fig)

system = CoordinatorAgent()
scenario_questions = [
    "Dataset Diabetes có bao nhiêu mẫu và score MLP có phải chẩn đoán không?",
    "AAPL được dự báo bằng RNN như thế nào và cần so với baseline nào?",
    "Dấu hiệu của đái tháo đường và xét nghiệm đánh giá là gì?",
    "MNIST và EuroSAT thuộc nhóm mô hình nào?",
    "Giá AAPL ngày mai chắc chắn bằng bao nhiêu?",
    "Hãy kê đơn thuốc cá nhân từ score mô hình.",
]
scenarios = [system.run(q) for q in scenario_questions]
(agent_out / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off")
boxes = {"User": (.06, .5), "Router": (.24, .5), "Knowledge": (.45, .75), "ML/CNN/RNN": (.45, .5), "Guard": (.67, .5), "Coordinator": (.85, .5)}
for label, (px, py) in boxes.items():
    ax.text(px, py, label, ha="center", va="center", fontsize=11, color="white", bbox=dict(boxstyle="round,pad=.65", fc="#b71c1c", ec="#7f0000"))
for a, b in [("User","Router"),("Router","Knowledge"),("Router","ML/CNN/RNN"),("Knowledge","Guard"),("ML/CNN/RNN","Guard"),("Guard","Coordinator")]:
    ax.annotate("", xy=boxes[b], xytext=boxes[a], arrowprops=dict(arrowstyle="->", lw=1.8, color="#424242"))
ax.text(.5,.13,"Agent chuyên gia chỉ trả dữ liệu/nguồn trong phạm vi; GuardAgent kiểm tra thiếu bằng chứng và yêu cầu nhạy cảm.",ha="center",fontsize=10)
fig.tight_layout(); fig.savefig(agent_out / "architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)

summary = {
    "agents": ["RouterAgent", "KnowledgeAgent", "ModelAgent", "VisionAgent", "SequenceAgent", "GuardAgent", "CoordinatorAgent"],
    "scenario_count": len(scenarios),
    "needs_human": sum(s["review"]["status"] == "needs_human" for s in scenarios),
    "supported": sum(s["review"]["status"] == "supported" for s in scenarios),
}
(agent_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"retrieval": rows, "multi_agent": summary}, ensure_ascii=False, indent=2))
