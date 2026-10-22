"""Ingestion, provenance, contracts and evaluation for Chapter 5.

The module deliberately uses the Python standard library plus NumPy so the
same artifacts can be rebuilt locally and on the lightweight Render service.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable


REQUIRED_DOCUMENT_FIELDS = {"id", "domain", "title", "url", "as_of", "text"}
REQUIRED_TRIPLE_FIELDS = {"subject", "relation", "object", "evidence"}
ALLOWED_DOMAINS = {"health", "finance"}
METHODS = {"B0": "BM25", "B1": "LSA", "B2": "Hybrid", "B3": "KG", "B4": "GraphRAG"}


def canonical_sha256(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_knowledge_base(documents: list[dict], triples: list[dict], queries: list[dict]) -> dict:
    """Reject missing schema fields, duplicate IDs and dangling provenance."""
    issues: list[str] = []
    document_ids = [row.get("id") for row in documents]
    if len(document_ids) != len(set(document_ids)):
        issues.append("Document id bị trùng.")
    for index, row in enumerate(documents):
        missing = REQUIRED_DOCUMENT_FIELDS - set(row)
        if missing:
            issues.append(f"Document {index} thiếu {sorted(missing)}.")
        if row.get("domain") not in ALLOWED_DOMAINS:
            issues.append(f"Document {row.get('id')} có domain không hợp lệ.")
        if not str(row.get("url", "")).startswith("https://"):
            issues.append(f"Document {row.get('id')} thiếu URL HTTPS.")
        if not str(row.get("text", "")).strip():
            issues.append(f"Document {row.get('id')} rỗng.")
    known = set(document_ids)
    triple_keys = set()
    for index, row in enumerate(triples):
        missing = REQUIRED_TRIPLE_FIELDS - set(row)
        if missing:
            issues.append(f"Triple {index} thiếu {sorted(missing)}.")
            continue
        key = (row["subject"], row["relation"], row["object"], row["evidence"])
        if key in triple_keys:
            issues.append(f"Triple bị trùng: {key}.")
        triple_keys.add(key)
        if row["evidence"] not in known:
            issues.append(f"Triple {index} trỏ tới evidence không tồn tại.")
    query_ids = [row.get("id") for row in queries]
    if len(query_ids) != len(set(query_ids)):
        issues.append("Query id bị trùng.")
    for row in queries:
        dangling = set(row.get("gold", [])) - known
        if dangling:
            issues.append(f"Query {row.get('id')} có gold không tồn tại: {sorted(dangling)}.")
        if row.get("group") not in {"direct", "paraphrase", "multi_hop", "out_of_scope"}:
            issues.append(f"Query {row.get('id')} có group không hợp lệ.")
    if issues:
        raise ValueError("\n".join(issues))
    return {
        "documents": len(documents),
        "triples": len(triples),
        "queries": len(queries),
        "domains": {name: sum(row["domain"] == name for row in documents) for name in sorted(ALLOWED_DOMAINS)},
        "document_hash": canonical_sha256(documents),
        "graph_hash": canonical_sha256(triples),
        "benchmark_hash": canonical_sha256(queries),
        "status": "valid",
    }


def chunk_documents(documents: list[dict], max_tokens: int = 45, overlap: int = 8) -> list[dict]:
    """Create stable overlapping chunks while preserving source provenance."""
    if max_tokens < 8 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("Cần max_tokens >= 8 và 0 <= overlap < max_tokens.")
    chunks = []
    step = max_tokens - overlap
    for document in documents:
        tokens = re.findall(r"\S+", document["text"])
        for number, start in enumerate(range(0, len(tokens), step)):
            selected = tokens[start:start + max_tokens]
            if not selected:
                break
            chunk_id = f"{document['id']}::c{number:03d}"
            text = " ".join(selected)
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document["id"],
                "domain": document["domain"],
                "title": document["title"],
                "url": document["url"],
                "as_of": document["as_of"],
                "token_start": start,
                "token_end": start + len(selected),
                "text": text,
                "sha256": canonical_sha256({"id": chunk_id, "text": text}),
            })
            if start + max_tokens >= len(tokens):
                break
    return chunks


def validate_request(payload: dict) -> dict:
    """Validate the public Chapter 5 request contract without silent coercion."""
    if not isinstance(payload, dict):
        raise ValueError("Request phải là JSON object.")
    allowed = {"question", "method", "top_k", "hops", "domain"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Trường không hỗ trợ: {sorted(unknown)}.")
    question = str(payload.get("question", "")).strip()
    if not 3 <= len(question) <= 500:
        raise ValueError("question phải dài từ 3 đến 500 ký tự.")
    method = payload.get("method", "B4")
    method = METHODS.get(method, method)
    if method not in METHODS.values():
        raise ValueError(f"method phải thuộc {list(METHODS)} hoặc {list(METHODS.values())}.")
    top_k = payload.get("top_k", 3)
    hops = payload.get("hops", 2)
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValueError("top_k phải là số nguyên từ 1 đến 10.")
    if not isinstance(hops, int) or not 0 <= hops <= 3:
        raise ValueError("hops phải là số nguyên từ 0 đến 3.")
    domain = payload.get("domain")
    if domain not in {None, *ALLOWED_DOMAINS}:
        raise ValueError("domain phải là health, finance hoặc null.")
    return {"question": question, "method": method, "top_k": top_k, "hops": hops, "domain": domain}


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / k


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold)) if gold else 0.0


def reciprocal_rank(retrieved: list[str], gold: list[str], k: int) -> float:
    relevant = set(gold)
    return next((1 / rank for rank, item in enumerate(retrieved[:k], 1) if item in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    relevant = set(gold)
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], 1) if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_ranking(retrieved: list[str], gold: list[str], k: int = 3) -> dict:
    return {
        f"Precision@{k}": precision_at_k(retrieved, gold, k),
        f"Recall@{k}": recall_at_k(retrieved, gold, k),
        f"MRR@{k}": reciprocal_rank(retrieved, gold, k),
        f"nDCG@{k}": ndcg_at_k(retrieved, gold, k),
        f"Complete@{k}": float(set(gold).issubset(retrieved[:k])) if gold else 0.0,
    }


def write_pipeline_artifacts(root: Path, documents: list[dict], triples: list[dict], queries: list[dict]) -> dict:
    """Write chunk and integrity artifacts used by notebook, report and deployment."""
    knowledge = root / "knowledge"
    output = root / "results" / "knowledge_rag"
    output.mkdir(parents=True, exist_ok=True)
    quality = validate_knowledge_base(documents, triples, queries)
    chunks = chunk_documents(documents)
    chunk_path = output / "chunks.jsonl"
    chunk_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in chunks) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "chapter": 5,
        "systems": METHODS,
        "quality": quality,
        "chunking": {"max_tokens": 45, "overlap": 8, "chunks": len(chunks)},
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in {
                "corpus": knowledge / "corpus.json",
                "graph": knowledge / "graph.json",
                "benchmark": knowledge / "benchmark.json",
                "chunks": chunk_path,
            }.items()
        },
    }
    (output / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
