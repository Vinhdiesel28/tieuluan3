from __future__ import annotations

import json
from pathlib import Path

from knowledge_rag import KnowledgeRAG, normalize

ROOT = Path(__file__).resolve().parents[1]


class RouterAgent:
    def route(self, question: str) -> list[str]:
        q = normalize(question)
        routes = []
        if any(x in q for x in ["dai thao duong", "diabetes", "a1c", "glucose", "suc khoe"]):
            routes += ["knowledge", "model"]
        if any(x in q for x in ["aapl", "co phieu", "gia", "rnn", "lstm", "return"]):
            routes += ["knowledge", "sequence"]
        if any(x in q for x in ["mnist", "eurosat", "anh", "cnn"]):
            routes += ["vision"]
        if any(x in q for x in ["khach hang", "retail", "mua lai"]):
            routes += ["sequence"]
        if not routes:
            routes = ["knowledge"]
        return list(dict.fromkeys(routes))


class ResultAgent:
    def __init__(self):
        self.summaries = {}
        for key in ["diabetes", "housing", "mnist", "eurosat", "stock", "customer"]:
            path = ROOT / "results" / key / "summary.json"
            if path.exists():
                self.summaries[key] = json.loads(path.read_text(encoding="utf-8"))

    def run(self, question: str, specialty: str) -> dict:
        q = normalize(question)
        if specialty == "vision":
            keys = [k for k in ["mnist", "eurosat"] if k in self.summaries]
        elif specialty == "sequence":
            keys = [k for k in ["stock", "customer"] if k in self.summaries]
        else:
            keys = [k for k in ["diabetes", "housing"] if k in self.summaries]
        if "aapl" in q or "co phieu" in q:
            keys = ["stock"]
        if "diabetes" in q or "dai thao duong" in q:
            keys = ["diabetes"]
        rows = []
        for key in keys:
            s = self.summaries[key]
            rows.append({"dataset": key, "task": s["task"], "samples": s["samples"], "splits": s["splits"], "source": s["source"]})
        return {"agent": specialty.title() + "Agent", "evidence": rows}


class KnowledgeAgent:
    def __init__(self):
        self.engine = KnowledgeRAG()

    def run(self, question: str) -> dict:
        domain = "health" if any(x in normalize(question) for x in ["dai thao duong", "diabetes", "a1c", "glucose"]) else "finance" if any(x in normalize(question) for x in ["aapl", "co phieu", "gia", "rnn"]) else None
        result = self.engine.answer(question, domain=domain)
        return {"agent": "KnowledgeAgent", "answer": result["answer"], "citations": result["citations"], "paths": result["paths"]}


class GuardAgent:
    def review(self, question: str, evidence_count: int) -> dict:
        q = normalize(question)
        blocked = any(x in q for x in ["chac chan", "ke don", "lieu thuoc", "mua hay ban", "co nen mua", "nen mua", "nen ban", "khuyen nghi dau tu"])
        return {
            "agent": "GuardAgent",
            "status": "needs_human" if blocked or evidence_count == 0 else "supported",
            "reason": "Yêu cầu cần chuyên gia hoặc dữ liệu tương lai." if blocked else "Có nguồn trong kho." if evidence_count else "Không có bằng chứng trong kho.",
        }


class CoordinatorAgent:
    def __init__(self):
        self.router = RouterAgent()
        self.results = ResultAgent()
        self.knowledge = KnowledgeAgent()
        self.guard = GuardAgent()

    def run(self, question: str) -> dict:
        routes = self.router.route(question)
        trace, citations, answer_parts = [{"agent": "RouterAgent", "routes": routes}], [], []
        for route in routes:
            if route == "knowledge":
                item = self.knowledge.run(question)
                trace.append(item)
                citations.extend(item["citations"])
                answer_parts.append(item["answer"])
            else:
                item = self.results.run(question, route)
                trace.append(item)
                if item["evidence"]:
                    answer_parts.append("Dữ liệu thực nghiệm: " + "; ".join(f"{r['dataset']} ({r['samples']} mẫu, tác vụ {r['task']})" for r in item["evidence"]))
        unique = {c["id"]: c for c in citations}
        guard = self.guard.review(question, len(unique))
        trace.append(guard)
        if guard["status"] != "supported":
            answer_parts.append("Hệ thống không đưa ra kết luận chắc chắn; cần người có chuyên môn hoặc dữ liệu phù hợp xác nhận.")
        return {
            "question": question,
            "answer": "\n\n".join(answer_parts)[:2400],
            "routes": routes,
            "trace": trace,
            "citations": list(unique.values()),
            "review": guard,
        }


if __name__ == "__main__":
    system = CoordinatorAgent()
    print(json.dumps(system.run("AAPL được dự báo bằng RNN như thế nào và có nên mua không?"), ensure_ascii=False, indent=2))
