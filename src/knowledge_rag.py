from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


class KnowledgeRAG:
    """BM25 + LSA + Knowledge Graph, chạy offline bằng NumPy."""

    def __init__(self, knowledge_dir: Path | None = None, dimensions: int = 8):
        folder = knowledge_dir or ROOT / "knowledge"
        self.documents = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
        self.triples = json.loads((folder / "graph.json").read_text(encoding="utf-8"))
        self.by_id = {d["id"]: d for d in self.documents}
        if any(t["evidence"] not in self.by_id for t in self.triples):
            raise ValueError("Graph chứa evidence id không tồn tại trong corpus.")
        self.doc_tokens = [tokenize(d["text"] + " " + d["title"]) for d in self.documents]
        self.lengths = np.array([len(x) for x in self.doc_tokens], dtype=float)
        self.avgdl = float(self.lengths.mean())
        df = Counter()
        for terms in self.doc_tokens:
            df.update(set(terms))
        n = len(self.documents)
        self.idf = {term: math.log(1 + (n - count + .5) / (count + .5)) for term, count in df.items()}
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.tfidf = self._matrix(self.doc_tokens)
        _, _, vt = np.linalg.svd(self.tfidf, full_matrices=False)
        self.projection = vt[: min(dimensions, len(vt))].T
        self.dense = self.tfidf @ self.projection
        self.alias = {}
        self.adj = defaultdict(list)
        for t in self.triples:
            s, o = normalize(t["subject"]), normalize(t["object"])
            self.alias[s] = t["subject"]
            self.alias[o] = t["object"]
            self.adj[s].append((o, t))
            self.adj[o].append((s, t))

    def _matrix(self, rows: list[list[str]]) -> np.ndarray:
        x = np.zeros((len(rows), len(self.vocab)), dtype=float)
        for i, terms in enumerate(rows):
            counts = Counter(terms)
            for term, count in counts.items():
                if term in self.vocab:
                    x[i, self.vocab[term]] = (1 + math.log(count)) * self.idf[term]
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)

    def _query_vector(self, query: str) -> np.ndarray:
        return self._matrix([tokenize(query)])[0]

    def bm25(self, query: str) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = []
        for doc, terms, dl in zip(self.documents, self.doc_tokens, self.lengths):
            tf, score = Counter(terms), 0.0
            for term in q:
                freq = tf.get(term, 0)
                if freq:
                    score += self.idf.get(term, 0) * freq * 2.5 / (freq + 1.5 * (.25 + .75 * dl / self.avgdl))
            if score > 0:
                scores.append((doc["id"], float(score)))
        return sorted(scores, key=lambda x: (-x[1], x[0]))

    def lsa(self, query: str) -> list[tuple[str, float]]:
        q = self._query_vector(query) @ self.projection
        denom = np.linalg.norm(self.dense, axis=1) * np.linalg.norm(q)
        scores = (self.dense @ q) / np.maximum(denom, 1e-12)
        rows = [(d["id"], float(s)) for d, s in zip(self.documents, scores) if s > 1e-12]
        return sorted(rows, key=lambda x: (-x[1], x[0]))

    def graph(self, query: str, hops: int = 2) -> tuple[list[tuple[str, float]], list[dict]]:
        nq = " " + normalize(query) + " "
        seeds = [node for node in self.alias if f" {node} " in nq]
        queue = deque((s, 0) for s in seeds)
        seen = set(seeds)
        evidence, paths = defaultdict(float), []
        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for nxt, triple in self.adj.get(node, []):
                score = 1 / (depth + 1)
                evidence[triple["evidence"]] += score
                paths.append({**triple, "hop": depth + 1})
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
        ranked = sorted(evidence.items(), key=lambda x: (-x[1], x[0]))
        return ranked, paths

    @staticmethod
    def rrf(rankings: list[tuple[list[tuple[str, float]], float]], constant: int = 60):
        scores = defaultdict(float)
        for ranking, weight in rankings:
            for rank, (doc_id, _) in enumerate(ranking, 1):
                scores[doc_id] += weight / (constant + rank)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    def retrieve(self, query: str, method: str = "GraphRAG", top_k: int = 3, hops: int = 2, domain: str | None = None):
        bm, ls = self.bm25(query), self.lsa(query)
        kg, paths = self.graph(query, hops)
        methods = {
            "BM25": bm,
            "LSA": ls,
            "KG": kg,
            "Hybrid": self.rrf([(bm, 1.0), (ls, .7)]),
            "GraphRAG": self.rrf([(bm, 1.0), (ls, .7), (kg, 1.2)]),
        }
        if method not in methods:
            raise ValueError(f"Phương pháp không hợp lệ: {method}")
        ranking = methods[method]
        if domain:
            ranking = [(doc_id, score) for doc_id, score in ranking if self.by_id[doc_id]["domain"] == domain]
        results = [{**self.by_id[doc_id], "score": round(score, 6)} for doc_id, score in ranking[:top_k]]
        selected = {d["id"] for d in results}
        return {"method": method, "query": query, "documents": results, "paths": [p for p in paths if p["evidence"] in selected]}

    def answer(self, query: str, method: str = "GraphRAG", top_k: int = 3, hops: int = 2, domain: str | None = None):
        result = self.retrieve(query, method, top_k, hops, domain)
        docs = result["documents"]
        if not docs:
            text = "Kho tri thức không có bằng chứng phù hợp để trả lời câu hỏi này."
        else:
            text = " ".join(d["text"] for d in docs)
        return {
            **result,
            "answer": text[:1800],
            "citations": [{"id": d["id"], "title": d["title"], "url": d["url"]} for d in docs],
            "notice": "Kết quả phục vụ học tập; prediction không thay thế chẩn đoán hoặc khuyến nghị đầu tư.",
        }


if __name__ == "__main__":
    engine = KnowledgeRAG()
    print(json.dumps(engine.answer("AAPL có baseline nào và RNN có phải khuyến nghị đầu tư không?"), ensure_ascii=False, indent=2))
