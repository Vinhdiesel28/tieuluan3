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
from knowledge_pipeline import METHODS, evaluate_ranking, write_pipeline_artifacts
from knowledge_rag import KnowledgeRAG
from multi_agent import CoordinatorAgent

rag_out = ROOT / "results" / "knowledge_rag"
agent_out = ROOT / "results" / "multi_agent"
rag_out.mkdir(parents=True, exist_ok=True)
agent_out.mkdir(parents=True, exist_ok=True)

documents = json.loads((ROOT / "knowledge/corpus.json").read_text(encoding="utf-8"))
triples = json.loads((ROOT / "knowledge/graph.json").read_text(encoding="utf-8"))
queries = json.loads((ROOT / "knowledge/benchmark.json").read_text(encoding="utf-8"))
manifest = write_pipeline_artifacts(ROOT, documents, triples, queries)
engine = KnowledgeRAG(ROOT / "knowledge")

rows, predictions, errors = [], [], []
for system, method in METHODS.items():
    answerable, latencies, ood_abstained = [], [], []
    for item in queries:
        start = time.perf_counter()
        answer = engine.answer(item["question"], method=method, top_k=3, hops=2)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        retrieved = [d["id"] for d in answer["documents"]]
        record = {
            "system": system,
            "method": method,
            "query_id": item["id"],
            "group": item["group"],
            "question": item["question"],
            "gold": item["gold"],
            "retrieved": retrieved,
            "scores": [d["score"] for d in answer["documents"]],
            "answer": answer["answer"],
            "citations": [c["id"] for c in answer["citations"]],
            "abstained": answer["abstained"],
            "reason": answer["reason"],
            "latency_ms": latency,
            "request_sha256": answer["request_sha256"],
            "corpus_sha256": answer["corpus_sha256"],
        }
        if item["gold"]:
            metrics = evaluate_ranking(retrieved, item["gold"], 3)
            record.update(metrics)
            answerable.append(metrics)
            missing = sorted(set(item["gold"]) - set(retrieved))
            if missing:
                errors.append({
                    "system": system, "method": method, "query_id": item["id"],
                    "group": item["group"], "missing_gold": ",".join(missing),
                    "retrieved": ",".join(retrieved), "error_type": "incomplete_retrieval",
                })
        else:
            ood_abstained.append(answer["abstained"])
            if not answer["abstained"]:
                errors.append({
                    "system": system, "method": method, "query_id": item["id"],
                    "group": item["group"], "missing_gold": "", "retrieved": ",".join(retrieved),
                    "error_type": "failed_to_abstain",
                })
        predictions.append(record)
    metric_names = ["Precision@3", "Recall@3", "MRR@3", "nDCG@3", "Complete@3"]
    rows.append({
        "System": system,
        "Method": method,
        **{name: float(np.mean([row[name] for row in answerable])) for name in metric_names},
        "OOD_AbstainRate": float(np.mean(ood_abstained)),
        "Latency_p50_ms": float(np.percentile(latencies, 50)),
        "Latency_p95_ms": float(np.percentile(latencies, 95)),
    })

with (rag_out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
with (rag_out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
    stream.write("\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
with (rag_out / "error_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
    fields = ["system", "method", "query_id", "group", "missing_gold", "retrieved", "error_type"]
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(errors)

# Ablation isolates graph depth and cut-off while keeping the benchmark fixed.
ablations = []
for hops in [0, 1, 2, 3]:
    for top_k in [1, 3, 5]:
        metrics = []
        for item in queries:
            if not item["gold"]:
                continue
            result = engine.retrieve(item["question"], method="GraphRAG", top_k=top_k, hops=hops)
            metrics.append(evaluate_ranking([d["id"] for d in result["documents"]], item["gold"], top_k))
        ablations.append({
            "hops": hops, "top_k": top_k,
            f"Recall@{top_k}": float(np.mean([row[f"Recall@{top_k}"] for row in metrics])),
            f"MRR@{top_k}": float(np.mean([row[f"MRR@{top_k}"] for row in metrics])),
            f"nDCG@{top_k}": float(np.mean([row[f"nDCG@{top_k}"] for row in metrics])),
        })
ablation_fields = sorted({key for row in ablations for key in row}, key=lambda x: (x not in {"hops", "top_k"}, x))
with (rag_out / "ablation.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=ablation_fields); writer.writeheader(); writer.writerows(ablations)

(rag_out / "benchmark.json").write_text(json.dumps({
    "manifest": manifest,
    "documents": len(engine.documents), "triples": len(engine.triples), "queries": len(queries),
    "rows": rows, "error_count": len(errors), "prediction_rows": len(predictions),
}, ensure_ascii=False, indent=2), encoding="utf-8")

x = np.arange(len(METHODS)); width = .21
fig, ax = plt.subplots(figsize=(10, 4.8))
for offset, metric, color in [(-width, "Recall@3", "#c62828"), (0, "MRR@3", "#ef5350"), (width, "nDCG@3", "#ff8a80")]:
    ax.bar(x + offset, [r[metric] for r in rows], width, label=metric, color=color)
ax.set_xticks(x, [f"{r['System']}\n{r['Method']}" for r in rows]); ax.set_ylim(0, 1.08)
ax.set_ylabel("Điểm trên 12 câu có nguồn"); ax.legend(ncol=3, loc="lower right"); ax.grid(axis="y", alpha=.2)
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
scenarios = [system.run(question) for question in scenario_questions]
(agent_out / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off")
boxes = {"User": (.06, .5), "Router": (.24, .5), "Knowledge": (.45, .75), "ML/CNN/RNN": (.45, .5), "Guard": (.67, .5), "Coordinator": (.85, .5)}
for label, (px, py) in boxes.items():
    ax.text(px, py, label, ha="center", va="center", fontsize=11, color="white", bbox=dict(boxstyle="round,pad=.65", fc="#b71c1c", ec="#7f0000"))
for source, target in [("User","Router"),("Router","Knowledge"),("Router","ML/CNN/RNN"),("Knowledge","Guard"),("ML/CNN/RNN","Guard"),("Guard","Coordinator")]:
    ax.annotate("", xy=boxes[target], xytext=boxes[source], arrowprops=dict(arrowstyle="->", lw=1.8, color="#424242"))
ax.text(.5,.13,"Agent chuyên gia chỉ trả dữ liệu/nguồn trong phạm vi; GuardAgent kiểm tra thiếu bằng chứng và yêu cầu nhạy cảm.",ha="center",fontsize=10)
fig.tight_layout(); fig.savefig(agent_out / "architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)

summary = {
    "agents": ["RouterAgent", "KnowledgeAgent", "ModelAgent", "VisionAgent", "SequenceAgent", "GuardAgent", "CoordinatorAgent"],
    "scenario_count": len(scenarios),
    "needs_human": sum(row["review"]["status"] == "needs_human" for row in scenarios),
    "supported": sum(row["review"]["status"] == "supported" for row in scenarios),
}
(agent_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"retrieval": rows, "errors": len(errors), "multi_agent": summary}, ensure_ascii=False, indent=2))
