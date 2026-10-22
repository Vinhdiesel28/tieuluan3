from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)
def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    nbf.write(nb, OUT / name)

setup = """from pathlib import Path
import json, sys
ROOT = Path.cwd()
if not (ROOT / 'knowledge').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
print('Project:', ROOT)"""

save("07_knowledge_graph_rag.ipynb", [
    md("# Chương 5 - Knowledge Graph và RAG\n\n<span style='color:#c62828'><b>PHẦN CẬP NHẬT MỚI.</b> Notebook đi từ dữ liệu, biểu diễn hàm, code từng thành phần đến kết quả benchmark. Các cell chạy từ trên xuống.</span>"),
    md("## 1. Khởi tạo\n\nNotebook dùng đúng corpus, graph và benchmark trong repository; không lấy nhãn test để xếp hạng."), code(setup),
    md("## 2. Ba lớp dữ liệu\n\nHai CSV Diabetes và AAPL là bảng quan sát. `corpus.json` là kho văn bản có nguồn. `graph.json` lưu quan hệ và `benchmark.json` chỉ dùng để đánh giá sau retrieval."),
    code("corpus=json.loads((ROOT/'knowledge/corpus.json').read_text(encoding='utf-8'))\ntriples=json.loads((ROOT/'knowledge/graph.json').read_text(encoding='utf-8'))\nqueries=json.loads((ROOT/'knowledge/benchmark.json').read_text(encoding='utf-8'))\nprint('Documents:',len(corpus),'| Triples:',len(triples),'| Queries:',len(queries))\nprint('Domains:',{d:sum(x['domain']==d for x in corpus) for d in ['health','finance']})"),
    md("## 3. Knowledge Graph dưới dạng triple\n\nMỗi cạnh có `subject`, `relation`, `object` và `evidence`. Trường evidence nối quan hệ về đúng document có URL."),
    code("for t in triples[:10]: print(f\"({t['subject']}) -[{t['relation']}]-> ({t['object']})  [{t['evidence']}]\")"),
    md("## 4. Hệ thống dưới dạng hợp các hàm\n\nVới câu hỏi $q$:  \\n$z=N(q)$ chuẩn hóa; $r_b=B(z)$ là BM25; $r_l=L(z)$ là LSA; $r_g=G(z,h)$ là graph retrieval tối đa $h$ bước; $r=F(r_b,r_l,r_g)$ dùng weighted RRF; cuối cùng $a=A(q,C(r))$ tạo đáp án từ context có nguồn.  \\nTách hàm giúp chỉ rõ lỗi nằm ở chuẩn hóa, retrieval, graph, hợp nhất hay generation."),
    md("## 5. Khởi tạo engine\n\nConstructor lập chỉ mục BM25, ma trận TF-IDF, phép chiếu SVD và adjacency list. Code đầy đủ nằm trong `src/knowledge_rag.py`."),
    code("from knowledge_rag import KnowledgeRAG\nengine=KnowledgeRAG(ROOT/'knowledge')\nprint('TF-IDF:',engine.tfidf.shape,'| Dense:',engine.dense.shape)"),
    md("## 6. So sánh năm bộ truy xuất\n\nBM25 dùng từ vựng; LSA dùng cosine trong không gian SVD; KG dùng entity và BFS; Hybrid hợp BM25+LSA; GraphRAG dùng đủ ba nhánh."),
    code("question='AAPL liên hệ với Apple, OHLCV và RNN như thế nào?'\nfor method in ['BM25','LSA','KG','Hybrid','GraphRAG']:\n    r=engine.retrieve(question,method=method,top_k=3,hops=2)\n    print(method,'->',[d['id'] for d in r['documents']])"),
    md("## 7. Đáp án có nguồn và đường graph\n\nĐầu ra giữ URL, document id và các cạnh đã đóng góp. Generation offline chỉ ghép bằng chứng, tránh phụ thuộc API trả phí."),
    code("answer=engine.answer('AAPL có baseline nào và RNN có phải khuyến nghị đầu tư không?')\nprint(answer['answer'])\nprint('Sources:',[x['id'] for x in answer['citations']])\nprint('Paths:',answer['paths'][:4])"),
    md("## 8. Kết quả benchmark\n\nMười hai câu có gold dùng Recall@3, MRR@3 và Complete@3. Ba câu ngoài phạm vi đo khả năng trả rỗng riêng; không trộn vào recall."),
    code("import pandas as pd\nfrom IPython.display import display,Image\ncomparison=pd.read_csv(ROOT/'results/knowledge_rag/comparison.csv')\ndisplay(comparison.round(4))\ndisplay(Image(filename=ROOT/'results/knowledge_rag/retrieval_comparison.png'))"),
    md("## 9. Nhận xét\n\nCorpus chỉ có 12 đoạn và benchmark do cùng dự án biên soạn. Điểm cao không chứng minh chuyển miền. Cơ chế từ chối còn yếu vì câu hỏi tương lai vẫn chia sẻ từ khóa với tài liệu lịch sử. Khi mở rộng cần corpus độc lập, người gán gold khác và đánh giá citation faithfulness.")
])

save("08_multi_agent_system.ipynb", [
    md("# Chương 6 - Hệ đa Agent\n\n<span style='color:#c62828'><b>PHẦN CẬP NHẬT MỚI.</b> Hệ thống điều phối Agent chuyên gia ML, CNN, RNN, Knowledge/RAG và GuardAgent.</span>"),
    md("## 1. Khởi tạo"), code(setup),
    md("## 2. Kiến trúc\n\nRouterAgent chọn chuyên gia theo mục đích câu hỏi. Các Agent chỉ trả bằng chứng trong phạm vi. GuardAgent kiểm tra thiếu nguồn hoặc yêu cầu nhạy cảm; CoordinatorAgent tổng hợp trace và citation."),
    code("from IPython.display import display,Image\ndisplay(Image(filename=ROOT/'results/multi_agent/architecture.png'))"),
    md("## 3. Hợp các hàm Agent\n\n$y = C(q, V(R(q)), K(q), S(q))$. Trong đó $R$ là router, $V$ gọi nhóm model/dataset, $K$ truy xuất tri thức, $S$ kiểm tra an toàn và $C$ phối hợp. Mỗi thành phần nhận và trả dictionary có schema rõ, nên trace có thể xem lại."),
    md("## 4. Code RouterAgent\n\nRouter dùng quy tắc có thể đọc được để demo. Hệ thống lớn có thể thay bằng classifier hoặc LLM structured output, nhưng vẫn phải giới hạn danh sách tool."),
    code("from multi_agent import RouterAgent\nrouter=RouterAgent()\nfor q in ['AAPL và RNN','Dấu hiệu diabetes','Ảnh MNIST dùng model nào?']:\n    print(q,'->',router.route(q))"),
    md("## 5. Coordinator và GuardAgent\n\nCoordinator không cho một Agent tự gọi công cụ tùy ý. GuardAgent chặn câu đòi giá chắc chắn, kê đơn hoặc khuyến nghị đầu tư; trạng thái `needs_human` được giữ trong kết quả."),
    code("from multi_agent import CoordinatorAgent\nsystem=CoordinatorAgent()\nr=system.run('AAPL được dự báo bằng RNN như thế nào và có nên mua không?')\nprint(r['answer'])\nprint('Routes:',r['routes'])\nprint('Review:',r['review'])\nprint('Sources:',[c['id'] for c in r['citations']])"),
    md("## 6. Sáu kịch bản đã chạy\n\nBa kịch bản có bằng chứng được hỗ trợ; ba kịch bản nhạy cảm hoặc thiếu dữ liệu được chuyển `needs_human`. Đây là chạy chức năng, không phải chứng nhận an toàn."),
    code("scenarios=json.loads((ROOT/'results/multi_agent/scenarios.json').read_text(encoding='utf-8'))\nfor s in scenarios:\n    print(s['review']['status'],'|',s['question'],'| routes=',s['routes'])"),
    md("## 7. Tích hợp web\n\nAPI `/api/agent` nhận `{question}` và trả answer, routes, trace, citations, review. Giao diện hiển thị đường điều phối và nguồn bên cạnh sáu bài toán dự đoán cũ. Render dùng cùng repository và không cần API key."),
    md("## 8. Giới hạn\n\nRouting dựa từ khóa, corpus nhỏ, chưa có bộ nhớ hội thoại dài hạn và chưa có sandbox công cụ ngoài. GuardAgent là kiểm tra quy tắc, không thay thế review của chuyên gia. Mọi Agent dùng cùng kết quả đã lưu để tránh mỗi nơi tự tạo một số liệu khác.")
])

print("Đã tạo notebook 07 và 08")
