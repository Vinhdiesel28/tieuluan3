from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)
def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    nbf.write(notebook, OUT / name)


setup = """from pathlib import Path
import inspect, json, sys
ROOT = Path.cwd()
if not (ROOT / 'knowledge').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
print('Project:', ROOT)"""

chapter5 = [
    md("# Chương 5 - Knowledge Base, Knowledge Graph và RAG\n\n<span style='color:#c62828'><b>PHẦN CODE ĐÃ BỔ SUNG.</b> Notebook trình bày đầy đủ ingestion, kiểm tra schema và provenance, chunking, năm cấu hình B0–B4, answer có citation, request contract, benchmark, ablation, phân tích lỗi và triển khai API. Chạy tuần tự từ trên xuống.</span>"),
    md("## 1. Khởi tạo\n\nMọi đường dẫn được suy ra từ thư mục dự án. Notebook dùng chung code trong `src/` với web để tránh một phiên bản demo khác phiên bản deploy."), code(setup),
    md("## 2. Nạp ba lớp dữ liệu\n\n`corpus.json` là kho tài liệu có URL; `graph.json` là các triple có evidence; `benchmark.json` chỉ dùng để đánh giá sau retrieval. Hai CSV Diabetes và AAPL là dữ liệu quan sát của các chương trước, không bị trộn vào gold evidence."),
    code("corpus=json.loads((ROOT/'knowledge/corpus.json').read_text(encoding='utf-8'))\ntriples=json.loads((ROOT/'knowledge/graph.json').read_text(encoding='utf-8'))\nqueries=json.loads((ROOT/'knowledge/benchmark.json').read_text(encoding='utf-8'))\nprint('Documents:',len(corpus),'| Triples:',len(triples),'| Queries:',len(queries))\nprint('Domains:',{d:sum(x['domain']==d for x in corpus) for d in ['health','finance']})"),
    md("## 3. Kiểm tra schema, liên kết và provenance\n\nHàm dưới đây từ chối document thiếu trường, ID trùng, URL không phải HTTPS, triple có evidence treo và query có gold ID không tồn tại. Ba hash khóa đúng snapshot dùng cho kết quả."),
    code("from knowledge_pipeline import validate_knowledge_base\nquality=validate_knowledge_base(corpus,triples,queries)\nquality"),
    md("### Code kiểm tra dữ liệu\n\nIn ra source thật đang được notebook và script thực nghiệm sử dụng."),
    code("print(inspect.getsource(validate_knowledge_base))"),
    md("## 4. Ingestion và chunking\n\nTài liệu được cắt theo cửa sổ 45 token, overlap 8 token. Mỗi chunk giữ `document_id`, URL, ngày nguồn, vị trí token và SHA-256. Corpus nhỏ vẫn chạy được ở cấp document; artifact chunk chứng minh pipeline có thể mở rộng mà không mất nguồn."),
    code("from knowledge_pipeline import chunk_documents\nchunks=chunk_documents(corpus,max_tokens=45,overlap=8)\nprint('Chunks:',len(chunks))\nprint(json.dumps(chunks[0],ensure_ascii=False,indent=2))"),
    md("## 5. Knowledge Graph và tính toàn vẹn evidence\n\nMỗi cạnh có `subject`, `relation`, `object`, `evidence`. Evidence phải là ID của document trong corpus, vì vậy một đường graph luôn truy ngược được về URL."),
    code("for t in triples[:10]: print(f\"({t['subject']}) -[{t['relation']}]-> ({t['object']}) [{t['evidence']}]\")"),
    md("## 6. Khởi tạo chỉ mục\n\nConstructor tạo thống kê BM25, ma trận TF-IDF, phép chiếu SVD tám chiều và adjacency list. Tất cả chạy offline bằng NumPy."),
    code("from knowledge_rag import KnowledgeRAG\nengine=KnowledgeRAG(ROOT/'knowledge')\nprint('TF-IDF:',engine.tfidf.shape,'| LSA dense:',engine.dense.shape,'| corpus hash:',engine.corpus_sha256[:16])"),
    md("## 7. B0 — BM25 sparse retrieval\n\nBM25 dùng tần suất từ, IDF và chuẩn hóa độ dài. Đây là baseline lexical, nhanh và giải thích được."),
    code("print(inspect.getsource(KnowledgeRAG.bm25))\nprint(engine.bm25('xét nghiệm glucose và A1C')[:3])"),
    md("## 8. B1 — LSA dense retrieval\n\nLSA chiếu TF-IDF qua SVD rồi tính cosine. Từ “dense” ở đây chỉ vector SVD dày, không phải embedding Transformer."),
    code("print(inspect.getsource(KnowledgeRAG.lsa))\nprint(engine.lsa('giá đầu phiên và khối lượng giao dịch')[:3])"),
    md("## 9. B3 — graph retrieval\n\nEntity linker dùng alias xác định; BFS duyệt tối đa `hops`. Hệ thống không đoán entity khi không có alias và không tạo cạnh mới lúc truy vấn."),
    code("print(inspect.getsource(KnowledgeRAG.graph))\nranked,paths=engine.graph('AAPL liên hệ Apple, OHLCV và RNN',hops=2)\nprint(ranked[:5]); print(json.dumps(paths[:3],ensure_ascii=False,indent=2))"),
    md("## 10. B2/B4 — Reciprocal Rank Fusion\n\nB2 hợp BM25 và LSA. B4 thêm KG. RRF dùng thứ hạng nên không cần coi điểm BM25, cosine và graph là cùng đơn vị."),
    code("print(inspect.getsource(KnowledgeRAG.rrf))\nquestion='AAPL liên hệ với Apple, OHLCV và RNN như thế nào?'\ncomparison=engine.compare(question,top_k=3,hops=2)\nfor system,result in comparison.items(): print(system,result['method'],'->',[d['id'] for d in result['documents']])"),
    md("## 11. Hợp hàm toàn pipeline\n\nVới câu hỏi $q$: normalize → B0/B1/B3 → RRF → top-k → grounded answer → citations/abstention. `compare` khóa cùng input, top-k và corpus cho cả B0–B4."),
    code("answer=engine.answer('AAPL có baseline nào và RNN có phải khuyến nghị đầu tư không?',method='B4')\nprint(answer['answer'])\nprint('Abstained:',answer['abstained'],'| Sources:',[x['id'] for x in answer['citations']])\nprint('Request hash:',answer['request_sha256'])"),
    md("## 12. Request contract và trường hợp biên\n\nAPI chỉ nhận năm trường allowlist, giới hạn độ dài, top-k, số hop và domain. Input lạ bị từ chối thay vì âm thầm bỏ qua."),
    code("from knowledge_pipeline import validate_request\nvalid=validate_request({'question':'Dấu hiệu diabetes là gì?','method':'B4','top_k':3,'hops':2,'domain':'health'})\nprint(valid)\nfor invalid in [{}, {'question':'x'}, {'question':'AAPL','method':'unknown'}, {'question':'AAPL','top_k':100}]:\n    try: validate_request(invalid)\n    except ValueError as exc: print('REJECT:',exc)"),
    md("## 13. Grounded answer, citation và từ chối\n\nAnswer chỉ lấy câu từ tài liệu đã truy hồi. Câu yêu cầu giá chắc chắn, kê đơn hoặc khuyến nghị mua/bán bị từ chối và không gắn citation giả."),
    code("for q in ['Xét nghiệm nào đánh giá diabetes?','Giá AAPL ngày mai chắc chắn bằng bao nhiêu?','Quỹ đạo Voyager là gì?']:\n    r=engine.answer(q,method='B4')\n    print('\\nQ:',q,'\\n',r['answer'],'\\nabstained=',r['abstained'],'citations=',[c['id'] for c in r['citations']])"),
    md("## 14. Metric retrieval đầy đủ\n\nNgoài Recall và MRR, notebook tính Precision@3, nDCG@3, Complete@3, OOD abstention cùng p50/p95 latency. Gold chỉ dùng sau khi ranking đã tạo."),
    code("import pandas as pd\nfrom IPython.display import display,Image\nmetrics=pd.read_csv(ROOT/'results/knowledge_rag/comparison.csv')\ndisplay(metrics.round(4))\ndisplay(Image(filename=ROOT/'results/knowledge_rag/retrieval_comparison.png'))"),
    md("## 15. Ablation số hop và top-k\n\nAblation chỉ thay một nhóm tham số, giữ nguyên corpus và câu hỏi. `hops=0` loại đóng góp graph; top-k 1/3/5 cho thấy đánh đổi precision–coverage."),
    code("ablation=pd.read_csv(ROOT/'results/knowledge_rag/ablation.csv')\ndisplay(ablation.round(4).fillna(''))"),
    md("## 16. Phân tích lỗi ở mức query\n\nMỗi lỗi ghi system, query ID, gold bị thiếu và danh sách đã truy hồi. File prediction JSONL còn giữ score, answer, citation, latency và hash để tái tính metric."),
    code("errors=pd.read_csv(ROOT/'results/knowledge_rag/error_analysis.csv')\nprint('Error rows:',len(errors))\ndisplay(errors.head(12))"),
    md("## 17. Hai case study theo miền\n\nCase sức khỏe giới hạn ở thông tin nguồn và nhắc không chẩn đoán. Case tài chính nối AAPL–OHLCV–RNN–baseline và từ chối biến dự báo thành lời khuyên đầu tư."),
    code("for domain,q in [('health','Type 2 phòng ngừa thế nào và nguồn tổng quan nói gì?'),('finance','Vì sao RNN AAPL cần so với Last value?')]:\n    r=engine.answer(q,method='B4',domain=domain)\n    print('\\nDOMAIN',domain,'\\n',r['answer'],'\\n',[c['url'] for c in r['citations']])"),
    md("## 18. Manifest và khả năng tái lập\n\nManifest ghi schema version, cấu hình chunk, hash và kích cỡ corpus/graph/benchmark/chunk. Đây là provenance của lần chạy, không phải mô tả nhập tay."),
    code("manifest=json.loads((ROOT/'results/knowledge_rag/pipeline_manifest.json').read_text(encoding='utf-8'))\nprint(json.dumps(manifest,ensure_ascii=False,indent=2))"),
    md("## 19. Triển khai\n\n`POST /api/rag` nhận request đúng schema và trả method, ranking, answer, paths, citations, abstention và hash. `POST /api/agent` dùng B4 như một công cụ trong hệ đa Agent. Cả hai không cần API key hoặc dịch vụ graph bên ngoài."),
    code("payload={'question':'AAPL gồm dữ liệu nào?','method':'B4','top_k':3,'hops':2,'domain':'finance'}\nprint('POST /api/rag')\nprint(json.dumps(payload,ensure_ascii=False,indent=2))"),
    md("## 20. Kết luận và giới hạn\n\nSo với phiên bản trước, code đã có ingestion/chunking, validation, provenance hash, B0–B4, request contract, grounded answer, abstention, metric mở rộng, ablation, prediction row-level và error analysis. Corpus 12 tài liệu vẫn nhỏ; kết quả không được suy rộng thành chất lượng trên MedQuAD, FinQA hoặc một hệ GraphRAG quy mô lớn.")
]
save("07_knowledge_graph_rag.ipynb", chapter5)

save("08_multi_agent_system.ipynb", [
    md("# Chương 6 - Hệ đa Agent\n\n<span style='color:#c62828'><b>PHẦN CẬP NHẬT MỚI.</b> Hệ thống điều phối Agent chuyên gia ML, CNN, RNN, Knowledge/RAG và GuardAgent.</span>"),
    md("## 1. Khởi tạo"), code(setup),
    md("## 2. Kiến trúc\n\nRouterAgent chọn chuyên gia theo mục đích câu hỏi. Các Agent chỉ trả bằng chứng trong phạm vi. GuardAgent kiểm tra thiếu nguồn hoặc yêu cầu nhạy cảm; CoordinatorAgent tổng hợp trace và citation."),
    code("from IPython.display import display,Image\ndisplay(Image(filename=ROOT/'results/multi_agent/architecture.png'))"),
    md("## 3. Hợp các hàm Agent\n\n$y = C(q, V(R(q)), K(q), S(q))$. Trong đó $R$ là router, $V$ gọi nhóm model/dataset, $K$ truy xuất tri thức, $S$ kiểm tra an toàn và $C$ phối hợp."),
    md("## 4. Code RouterAgent"), code("from multi_agent import RouterAgent\nrouter=RouterAgent()\nfor q in ['AAPL và RNN','Dấu hiệu diabetes','Ảnh MNIST dùng model nào?']:\n    print(q,'->',router.route(q))"),
    md("## 5. Coordinator và GuardAgent"), code("from multi_agent import CoordinatorAgent\nsystem=CoordinatorAgent()\nr=system.run('AAPL được dự báo bằng RNN như thế nào và có nên mua không?')\nprint(r['answer'])\nprint('Routes:',r['routes'])\nprint('Review:',r['review'])\nprint('Sources:',[c['id'] for c in r['citations']])"),
    md("## 6. Sáu kịch bản đã chạy"), code("scenarios=json.loads((ROOT/'results/multi_agent/scenarios.json').read_text(encoding='utf-8'))\nfor s in scenarios:\n    print(s['review']['status'],'|',s['question'],'| routes=',s['routes'])"),
    md("## 7. Tích hợp web\n\nAPI `/api/agent` trả answer, routes, trace, citations và review. KnowledgeAgent dùng cùng code B4 của `/api/rag`; Render không cần API key."),
    md("## 8. Giới hạn\n\nRouting dựa từ khóa, corpus nhỏ, chưa có bộ nhớ hội thoại dài hạn và chưa có sandbox công cụ ngoài. GuardAgent là kiểm tra quy tắc, không thay thế review của chuyên gia.")
])

print("Built notebooks 07 and 08")
