"""Các chương cập nhật màu đỏ cho Tiểu luận 03."""
from __future__ import annotations

import csv
import json


def append_agent_sections(root, page, p, table, fig, code):
    corpus = json.loads((root / "knowledge/corpus.json").read_text(encoding="utf-8"))
    graph = json.loads((root / "knowledge/graph.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "knowledge/benchmark.json").read_text(encoding="utf-8"))
    with (root / "results/knowledge_rag/comparison.csv").open(encoding="utf-8-sig") as stream:
        scores = list(csv.DictReader(stream))
    with (root / "results/knowledge_rag/ablation.csv").open(encoding="utf-8-sig") as stream:
        ablation = list(csv.DictReader(stream))
    manifest = json.loads((root / "results/knowledge_rag/pipeline_manifest.json").read_text(encoding="utf-8"))
    error_rows = sum(1 for _ in (root / "results/knowledge_rag/error_analysis.csv").open(encoding="utf-8-sig")) - 1
    agents = json.loads((root / "results/multi_agent/summary.json").read_text(encoding="utf-8"))

    page("CHƯƠNG 5. KNOWLEDGE BASE, KNOWLEDGE GRAPH VÀ RETRIEVAL-AUGMENTED GENERATION")
    p("PHẦN CẬP NHẬT MỚI CỦA TIỂU LUẬN 03.")
    p("Chương này mở rộng quyển tiểu luận từ mô hình dự đoán sang hệ thống trả lời có bằng chứng. Mô hình ML, CNN hoặc RNN ánh xạ đầu vào thành nhãn hay giá trị; RAG nhận một câu hỏi, truy hồi các đoạn liên quan rồi tạo câu trả lời kèm nguồn. Knowledge Graph bổ sung các thực thể và quan hệ để truy vấn không chỉ phụ thuộc vào từ khóa.")
    p("Thiết kế được biểu diễn như hợp hàm: answer = compose(review, generate, retrieve, normalize)(question). Mỗi hàm có đầu vào, đầu ra và trách nhiệm riêng. Nhờ đó có thể thay BM25 bằng LSA, thêm mở rộng đồ thị hoặc đổi cách tổng hợp mà không làm thay đổi toàn bộ hệ thống.")
    table(["Thành phần", "Đầu vào", "Đầu ra", "Mục đích"], [
        ["Normalize", "Câu hỏi Unicode", "Token chuẩn hóa", "Giảm khác biệt chữ hoa và dấu"],
        ["Retriever", "Token và chỉ mục", "Danh sách tài liệu", "Tìm bằng chứng phù hợp"],
        ["Knowledge Graph", "Thực thể hạt giống", "Đường đi và evidence", "Mở rộng quan hệ nhiều bước"],
        ["Generator", "Câu hỏi và bằng chứng", "Câu trả lời có trích dẫn", "Không trả lời ngoài dữ liệu"],
    ], [100, 115, 120, 145])

    page("5.1. Mở đầu")
    p("Knowledge Base (KB) là kho tri thức có cấu trúc dùng để lưu tài liệu, nguồn, ngày cập nhật và các phát biểu có thể truy vết. Knowledge Graph (KG) là một cách tổ chức KB dưới dạng mạng thực thể–quan hệ. Retrieval-Augmented Generation (RAG) dùng truy hồi để đưa bằng chứng từ KB/KG vào quá trình trả lời. Ba lớp này bổ sung cho mô hình dự đoán ở Chương 2–4: hệ thống vừa đưa ra kết quả, vừa chỉ ra dữ liệu nào hỗ trợ kết quả đó.")
    p("Mục tiêu của chương là xây dựng pipeline có thể chạy lại trên hai miền đã dùng trong quyển: bệnh đái tháo đường và chuỗi AAPL. Bài so sánh năm cấu hình B0–B4, kiểm tra citation, khả năng từ chối câu ngoài phạm vi, phân tích lỗi và cung cấp API triển khai. Phạm vi là minh họa học thuật; câu trả lời sức khỏe không thay thế chẩn đoán và nội dung tài chính không phải khuyến nghị đầu tư.")

    page("5.2. Knowledge Base và Knowledge Graph")
    p("Knowledge Base của bài gồm tài liệu, metadata nguồn và các bộ ba quan hệ. Mỗi tài liệu có id, domain, title, url, as_of và text. Knowledge Graph biểu diễn tri thức bằng bộ ba (subject, predicate, object). Subject và object là thực thể; predicate mô tả quan hệ có hướng. Mỗi cạnh còn mang trường evidence trỏ về một tài liệu trong corpus, nhờ đó quan hệ dùng để trả lời luôn có thể truy vết.")
    p("Đồ thị được lưu dạng property graph trong JSON. Khi nạp dữ liệu, mã tạo danh sách kề hai chiều để phục vụ tìm kiếm. Với câu hỏi có thực thể khớp alias, BFS duyệt tối đa hai hop. Điểm một cạnh tại độ sâu d là 1/(d+1), vì quan hệ xa hơn thường ít trực tiếp hơn.")
    table(["Thống kê", "Giá trị", "Ý nghĩa"], [
        ["Tài liệu", str(len(corpus)), "6 sức khỏe và 6 tài chính"],
        ["Bộ ba", str(len(graph)), "Quan hệ đều có evidence hợp lệ"],
        ["Miền", "health, finance", "Khớp hai CSV được nộp"],
        ["Số hop", "1–2", "Giới hạn nhiễu khi mở rộng"],
    ], [120, 110, 250])
    code('''queue = deque((seed, 0) for seed in seeds)
while queue:
    node, depth = queue.popleft()
    if depth >= hops:
        continue
    for next_node, triple in adjacency[node]:
        evidence[triple["evidence"]] += 1 / (depth + 1)
        queue.append((next_node, depth + 1))''')

    page("5.3. Retrieval-Augmented Generation")
    p("BM25 là phương pháp truy hồi từ vựng. Điểm tăng khi từ của câu hỏi xuất hiện nhiều trong tài liệu, nhưng được điều chỉnh theo độ dài tài liệu và độ hiếm IDF. Bài dùng k1=1,5 và b=0,75. Ưu điểm là nhanh, dễ giải thích và phù hợp corpus nhỏ; nhược điểm là khó nối các cách diễn đạt khác nhau.")
    p("LSA xây ma trận TF-IDF tài liệu–từ, chuẩn hóa từng hàng và phân rã SVD. Câu hỏi được chiếu vào không gian tám chiều rồi xếp hạng theo cosine similarity. Đây là mô hình dense cổ điển chạy hoàn toàn offline bằng NumPy. Nó không được mô tả như embedding ngôn ngữ hiện đại và không dùng dịch vụ bên ngoài.")
    code('''tfidf = normalize_rows(term_frequency * inverse_document_frequency)
_, _, vt = np.linalg.svd(tfidf, full_matrices=False)
projection = vt[:8].T
dense_docs = tfidf @ projection
score = cosine(dense_docs, query_tfidf @ projection)''')
    p("Hai danh sách BM25 và LSA được kết hợp bằng Reciprocal Rank Fusion: RRF(d)=Σ w/(60+rank). Phương pháp dùng thứ hạng thay cho thang điểm gốc nên tránh phải ép điểm BM25, cosine và đồ thị về cùng đơn vị.")

    page("5.3.1. GraphRAG và các baseline B0–B4")
    p("GraphRAG trong bài là pipeline xác định, không gọi mô hình ngôn ngữ lớn. Hệ thống tính ba ranking BM25, LSA và KG; hợp nhất bằng RRF với trọng số lần lượt 1,0; 0,7; 1,2; lấy top-3; rồi ghép các đoạn bằng chứng và danh sách URL. Cách làm này đủ để đánh giá truy hồi, dễ chạy trên máy cá nhân và tránh phát sinh câu không có trong nguồn.")
    p("Một truy vấn trực tiếp có thể được BM25 giải quyết. Truy vấn nhiều bước như mối quan hệ giữa AAPL, OHLCV, RNN và giới hạn đầu tư nhận thêm đường đi từ graph. Nếu không có tài liệu, answer trả thông báo thiếu bằng chứng. Trong miền sức khỏe và tài chính, câu trả lời luôn kèm lưu ý về phạm vi học thuật.")
    code('''ranking = rrf([
    (bm25(query), 1.0),
    (lsa(query), 0.7),
    (graph(query, hops=2), 1.2),
])
documents = ranking[:3]
answer = compose_notice(compose_citations(documents))''')
    table(["Phương pháp", "Tín hiệu", "Điểm mạnh", "Giới hạn"], [
        ["BM25", "Từ vựng", "Nhanh, giải thích được", "Nhạy cách dùng từ"],
        ["LSA", "Không gian SVD", "Bắt quan hệ tiềm ẩn đơn giản", "Corpus nhỏ làm vector kém ổn định"],
        ["KG", "Thực thể và cạnh", "Có đường đi", "Phụ thuộc alias và độ phủ graph"],
        ["GraphRAG", "Ba ranking", "Kết hợp nguồn và quan hệ", "Không mặc nhiên tốt nhất"],
    ], [85, 105, 145, 145])

    page("5.4. Case study KB/RAG cho bệnh đái tháo đường")
    p("Case study thứ nhất dùng lại CDC Diabetes để nối mô hình phân loại ở Chương 2 với lớp tri thức. CSV cung cấp 21 biến khảo sát và nhãn Diabetes_012; KB sức khỏe gồm sáu tài liệu từ WHO, CDC, NIDDK và metadata của bài. Graph liên kết các khái niệm như diabetes, blood glucose, BMI, physical activity và risk factor. Hệ thống trả đoạn bằng chứng và URL, không suy diễn một hồ sơ thành chẩn đoán y khoa.")
    table(["Thành phần", "Kích cỡ", "Output / nhãn", "Nguồn và vai trò"], [
        ["CDC Diabetes", "253.680 × 22", "Diabetes_012; bài gộp 1/2 thành dương", "BRFSS 2015; dữ liệu bảng sức khỏe"],
        ["Health KB", "6 tài liệu", "Đoạn bằng chứng và URL", "WHO, CDC, NIDDK và metadata bài"],
        ["Health graph", "Các triple miền health", "subject–predicate–object", "Evidence liên kết tài liệu sức khỏe"],
        ["Health queries", "Direct, paraphrase, multi-hop, OOD", "Gold document ids", "Đánh giá truy hồi và từ chối"],
    ], [105, 95, 145, 135])
    p("Ví dụ đầu vào là câu hỏi về mối liên hệ giữa vận động, BMI và nguy cơ tiểu đường. Pipeline chuẩn hóa câu hỏi, tìm tài liệu liên quan, mở rộng thực thể trên graph rồi trả câu dựa trên bằng chứng. Với câu yêu cầu kê đơn hoặc khẳng định chẩn đoán, Guard thêm cảnh báo và chuyển trạng thái cần người có chuyên môn xem xét.")

    page("5.5. Case study KB/RAG cho AAPL và báo cáo doanh nghiệp")
    p("Case study thứ hai dùng AAPL 2015–2025 để nối mô hình RNN ở Chương 4 với tri thức tài chính. CSV gồm Date, Open, High, Low, Close, Adj Close và Volume. KB tài chính gồm sáu tài liệu về dữ liệu OHLCV, cách tạo window, dự báo one-step, giới hạn của mô hình và nguồn Apple Investor Relations. KG liên kết AAPL, market data, RNN, forecast và investment risk.")
    table(["Thành phần", "Kích cỡ", "Output", "Vai trò"], [
        ["AAPL 2015–2025", "2.766 × 7", "Close phiên kế tiếp", "Chuỗi thời gian cho RNN"],
        ["Finance KB", "6 tài liệu", "Đoạn bằng chứng và URL", "Giải thích dữ liệu, mô hình và giới hạn"],
        ["Finance graph", "Các triple miền finance", "Đường đi 1–2 hop", "Nối AAPL–OHLCV–RNN–risk"],
        ["Finance queries", "Direct, paraphrase, multi-hop, OOD", "Gold document ids", "Đánh giá truy hồi"],
    ], [110, 95, 145, 130])
    p("Câu trả lời có thể giải thích dữ liệu nào được dùng để dự báo và vì sao đầu ra chỉ là ước lượng một bước. Nó không đưa lệnh mua/bán. Đây là khác biệt có chủ ý so với bài mẫu dùng FinQA: bài này giữ AAPL để thống nhất dataset, mô hình và kết quả đã trình bày ở Chương 4.")

    page("5.6. Hợp các hàm và contract dùng chung")
    p("Phần code được bổ sung sau khi đối chiếu bài mẫu gồm kiểm tra schema, phát hiện ID trùng, URL không hợp lệ, dangling evidence và gold document không tồn tại. Pipeline cắt tài liệu thành cửa sổ 45 token, overlap 8; mỗi chunk giữ document_id, URL, ngày nguồn, vị trí token và SHA-256.")
    table(["Artifact", "Số lượng / cấu hình", "Kiểm soát"], [
        ["Documents", str(manifest["quality"]["documents"]), "ID duy nhất, domain và HTTPS URL"],
        ["Triples", str(manifest["quality"]["triples"]), "Evidence phải tồn tại trong corpus"],
        ["Queries", str(manifest["quality"]["queries"]), "Gold ID và group hợp lệ"],
        ["Chunks", str(manifest["chunking"]["chunks"]), "45 token, overlap 8, hash từng chunk"],
        ["Snapshot", manifest["quality"]["document_hash"][:16] + "…", "Hash corpus, graph, benchmark và chunk"],
    ], [95, 145, 240])
    code('''quality = validate_knowledge_base(corpus, triples, queries)
chunks = chunk_documents(corpus, max_tokens=45, overlap=8)
manifest = {
    "schema_version": "1.0.0",
    "quality": quality,
    "files": hash_every_input_and_chunk_file(),
}''')
    p("Các hash không chứng minh nguồn bên ngoài đúng, nhưng xác định chính xác snapshot nào đã sinh bảng kết quả. File pipeline_contract.schema.json và hàm validate_request giới hạn question, method, top_k, hops và domain; trường lạ bị từ chối.")

    page("5.6.1. Mã nguồn và cách chạy notebook")
    p("Notebook 07 hiện có 20 code cell, giải thích từ ingestion đến deployment. Các cell đã có output, có thể Restart Kernel rồi Run All. Mã nghiệp vụ nằm trong src/knowledge_pipeline.py và src/knowledge_rag.py để notebook, web và script đánh giá dùng cùng implementation.")
    code('''from src.knowledge_rag import KnowledgeRAG

rag = KnowledgeRAG()
result = rag.answer(
    "AAPL được dự báo bằng dữ liệu nào và có phải khuyến nghị đầu tư không?",
    method="GraphRAG", top_k=3, hops=2,
)
for source in result["citations"]:
    print(source["title"], source["url"])''')
    p("Việc giữ nguồn trong cấu trúc answer giúp giao diện hiển thị link thật thay vì chỉ in một đoạn văn. Cùng câu hỏi và cùng corpus cho kết quả xác định, thuận lợi cho chấm bài và phân tích lỗi.")

    page("5.7. Cầu nối sang hệ đa tác tử")
    p("Pipeline Chương 5 được đóng gói thành KnowledgeAgent để Chương 6 có thể gọi như một công cụ. Router chỉ chọn miền và tuyến xử lý; KnowledgeAgent nhận request đã kiểm tra, chạy B4 GraphRAG và trả answer, citations, graph paths, request hash cùng corpus hash. GuardAgent sau đó kiểm tra số citation và các từ khóa rủi ro trước khi Coordinator tổng hợp.")
    table(["Bước", "Contract đầu vào", "Contract đầu ra"], [
        ["Router", "question", "health / finance / knowledge / model"],
        ["KnowledgeAgent", "question, domain, top_k, hops", "answer, citations, paths, hashes"],
        ["GuardAgent", "question, evidence count", "supported / needs_human và lý do"],
        ["Coordinator", "routes, evidence, review", "final answer và trace"],
    ], [105, 180, 195])
    p("Cách nối bằng contract giúp notebook, web API và hệ đa Agent dùng cùng một implementation. Trace chỉ mô tả luồng phần mềm và bằng chứng được dùng; nó không được diễn giải thành lời giải thích nhân quả của mô hình.")

    page("5.8. Đánh giá và tiểu kết")
    rows = []
    for row in scores:
        rows.append([row["System"]+" "+row["Method"], f'{float(row["Precision@3"]):.4f}', f'{float(row["Recall@3"]):.4f}', f'{float(row["MRR@3"]):.4f}', f'{float(row["nDCG@3"]):.4f}', f'{float(row["OOD_AbstainRate"]):.4f}', f'{float(row["Latency_p95_ms"]):.3f}'])
    table(["System", "P@3", "R@3", "MRR", "nDCG", "OOD", "p95 ms"], rows, [105, 58, 58, 58, 58, 65, 78])
    fig(root / "results/knowledge_rag/retrieval_comparison.png", "So sánh năm phương pháp trên benchmark 15 câu.", maxheight=260)
    p("B4 GraphRAG đạt Recall@3 cao nhất 0,9375; B0 và B2 đạt 0,9167. B0/B2 có MRR cao hơn B4, cho thấy thêm graph cải thiện độ phủ nhưng không luôn đưa tài liệu đúng lên hạng đầu. Cả năm hệ thống đạt OOD abstention 1,0 sau lớp guardrail; đây là kết quả trên ba câu OOD đã định nghĩa, không phải chứng nhận an toàn tổng quát.")

    page("5.8.1. Ablation và phân tích lỗi")
    selected_ablation = [row for row in ablation if row["top_k"] == "3"]
    table(["Graph hops", "Top-k", "Recall@3", "MRR@3", "nDCG@3"], [[row["hops"],row["top_k"],f'{float(row["Recall@3"]):.4f}',f'{float(row["MRR@3"]):.4f}',f'{float(row["nDCG@3"]):.4f}'] for row in selected_ablation],[90,75,100,100,115])
    p(f"Ablation thay số hop 0–3 và top-k 1/3/5, giữ nguyên corpus và query. File prediction có {len(scores)*len(benchmark)} dòng; error_analysis.csv có {error_rows} trường hợp thiếu ít nhất một gold document. Mỗi dòng giữ query ID, retrieved IDs, score, citation, latency, request hash và corpus hash để có thể tái tính metric.")
    p("Với corpus nhỏ, thay đổi hop có thể đưa thêm evidence đúng hoặc đẩy một tài liệu từ vựng khỏi top-k. Vì vậy báo cáo đồng thời Recall, MRR và nDCG; chỉ dùng một metric có thể che khuất sự thay đổi thứ hạng.")

    page("5.8.2. Nhận xét, sai số và giới hạn")
    p("Sai số thứ nhất đến từ alias: graph chỉ khởi động khi câu hỏi chứa tên thực thể đã biết. Sai số thứ hai đến từ corpus nhỏ, khiến SVD có ít ngữ cảnh và một tài liệu có thể chứa nhiều khái niệm. Sai số thứ ba là phép ghép bằng chứng: RRF cải thiện độ phủ nhưng tài liệu được thêm từ graph có thể làm mất một tài liệu từ vựng ở top-3.")
    p("Hệ thống hiện là extractive RAG: câu trả lời ghép nội dung nguồn, chưa có mô hình sinh. Vì vậy nó ít hallucination nhưng diễn đạt kém tự nhiên hơn LLM. Nếu thay bằng mô hình sinh cần thêm prompt grounding, kiểm tra citation entailment, chống prompt injection và đánh giá factuality.")
    table(["Rủi ro", "Biện pháp hiện tại", "Hướng tiếp theo"], [
        ["Trả lời ngoài phạm vi", "Thông báo không có bằng chứng", "Ngưỡng score được hiệu chỉnh"],
        ["Quan hệ sai", "Mỗi triple bắt buộc evidence id", "Kiểm duyệt graph và versioning"],
        ["Nguồn lỗi thời", "Có as_of và URL", "Lịch làm mới và snapshot"],
        ["Diễn giải quá mức", "Guard notice sức khỏe/tài chính", "Human review theo rủi ro"],
    ], [105, 175, 200])

    page("5.9. Mẫu dữ liệu và đường dẫn hồ sơ")
    p("Toàn bộ hồ sơ thực nghiệm được lưu kèm repository để có thể đối chiếu bảng trong báo cáo. CSV gốc phục vụ mô hình dự đoán; corpus, graph và benchmark phục vụ RAG; các artifact kết quả giữ dự đoán theo từng câu, ablation, lỗi và manifest hash.")
    table(["Hồ sơ", "Đường dẫn", "Nội dung"], [
        ["Dữ liệu Diabetes", "data/diabetes.csv", "253.680 dòng dữ liệu bảng"],
        ["Dữ liệu AAPL", "data/AAPL_2015_2025.csv", "2.766 phiên OHLCV"],
        ["Corpus / KG / benchmark", "knowledge/*.json", "Tài liệu, triple và gold query"],
        ["Pipeline artifacts", "results/knowledge_rag/", "Predictions, ablation, errors, chunks, manifest"],
        ["Notebook", "notebooks/07_knowledge_graph_rag.ipynb", "20 code cell đã chạy và có output"],
    ], [115, 190, 175])
    code('''document = {"id": "...", "domain": "health|finance",
            "title": "...", "url": "https://...", "as_of": "...", "text": "..."}
triple = {"subject": "...", "predicate": "...", "object": "...",
          "evidence": "document_id"}
query = {"id": "...", "question": "...", "gold": ["document_id"],
         "group": "direct|paraphrase|multi_hop|ood"}''')

    page("5.10. Triển khai ngắn")
    p("Web bổ sung POST /api/rag bên cạnh POST /api/agent. API RAG nhận question, B0–B4 hoặc tên phương pháp, top_k, hops và domain. Response trả ranking, graph paths, grounded answer, citations, abstention, reason, request SHA-256 và corpus SHA-256.")
    code('''payload = validate_request(request.get_json())
question = payload.pop("question")
result = engine.answer(question, **payload)
return jsonify(result)''')
    p("Bộ triển khai không nhận đường dẫn file, pickle hoặc tool tùy ý từ người dùng. Corpus và graph được đóng gói trong repository; inference chỉ dùng NumPy và JSON nên phù hợp Render Free. Câu hỏi rỗng, trường lạ, top-k/hops vượt giới hạn hoặc domain sai nhận HTTP 400.")

    page("CHƯƠNG 6. HỆ ĐA AGENT")
    p("PHẦN CẬP NHẬT MỚI CỦA TIỂU LUẬN 03.")
    p("Hệ đa Agent chia một yêu cầu thành các vai trò có trách nhiệm rõ ràng. Coordinator nhận câu hỏi và điều phối. Router xác định miền. KnowledgeAgent gọi GraphRAG. ModelAgent, VisionAgent và SequenceAgent đọc metadata kết quả của Chương 2–4. GuardAgent kiểm tra bằng chứng và phát hiện yêu cầu cần con người xác nhận.")
    p("Các agent trong bài là thành phần phần mềm chạy tuần tự và xác định. Chúng không tự sửa code, không tự truy cập Internet và không tự thực hiện giao dịch hay hành động y tế. Cách giới hạn này phù hợp với dữ liệu và mục tiêu học thuật của hệ thống.")
    fig(root / "results/multi_agent/architecture.png", "Kiến trúc điều phối bảy agent.", maxheight=330)

    page("6.1. Router và các Agent chuyên trách")
    table(["Agent", "Nhiệm vụ", "Dữ liệu sử dụng"], [
        ["RouterAgent", "Chọn một hoặc nhiều tuyến", "Từ khóa đã chuẩn hóa"],
        ["KnowledgeAgent", "Truy hồi và trả bằng chứng", "Corpus và Knowledge Graph"],
        ["ModelAgent", "Mô tả ML dữ liệu bảng", "Diabetes, housing summaries"],
        ["VisionAgent", "Mô tả CNN", "MNIST, EuroSAT summaries"],
        ["SequenceAgent", "Mô tả RNN", "AAPL, retail summaries"],
        ["GuardAgent", "Quyết định supported/needs_human", "Số citation và từ khóa rủi ro"],
        ["CoordinatorAgent", "Tổng hợp trace và answer", "Output của các agent"],
    ], [105, 180, 195])
    p("Router có thể chọn nhiều tuyến. Ví dụ câu hỏi về AAPL và RNN gọi KnowledgeAgent để lấy nguồn, đồng thời gọi SequenceAgent để lấy thống kê thực nghiệm. Điều này khác một chuỗi cố định, vì nhánh được quyết định từ nội dung câu hỏi.")

    page("6.2. Hợp hàm và luồng điều phối")
    code('''def run(question):
    routes = router.route(question)
    evidence = [dispatch(route, question) for route in routes]
    review = guard.review(question, count_citations(evidence))
    return coordinator.compose(question, routes, evidence, review)''')
    p("Trace giữ tên agent, tuyến được chọn, evidence và kết quả Guard. Đây là log giải thích quyết định ở mức hệ thống. Nó không chứng minh mô hình đúng về mặt nhân quả, nhưng cho biết thành phần nào đã đóng góp vào câu trả lời và vì sao hệ thống yêu cầu người kiểm tra.")
    p("Hàm Guard đánh dấu needs_human khi không có citation hoặc câu hỏi đòi kê đơn, kết luận chắc chắn hay khuyến nghị mua/bán. Quy tắc này có chủ ý đơn giản và có thể kiểm tra. Trong ứng dụng thật cần policy theo quyền hạn, dữ liệu riêng tư, audit log và cơ chế phê duyệt.")

    page("6.3. Kết quả kịch bản đa Agent")
    table(["Chỉ số", "Kết quả"], [
        ["Số Agent", str(len(agents["agents"]))],
        ["Danh sách", ", ".join(agents["agents"])],
        ["Kịch bản", str(agents["scenario_count"])],
        ["Supported", str(agents["supported"])],
        ["Needs human", str(agents["needs_human"])],
    ], [120, 360])
    p("Sáu kịch bản bao phủ dữ liệu bảng, ảnh, chuỗi, truy hồi tri thức, câu hỏi rủi ro và câu ngoài phạm vi. Ba trường hợp được trả ở trạng thái supported; ba trường hợp cần người xác nhận. Tỷ lệ needs_human không phải accuracy, mà phản ánh policy thận trọng của Guard trên tập tình huống đã định nghĩa.")
    p("Notebook 08 hiển thị từng bước route → specialist → guard → coordinator. Output được lưu trong notebook để người chấm có thể đọc mà không cần khởi động server. File results/multi_agent/scenarios.json giữ toàn bộ trace ở dạng máy đọc.")

    page("6.4. Tích hợp web và triển khai")
    p("Web Flask được bổ sung API POST /api/agent. Payload gồm question; server gọi CoordinatorAgent và trả answer, routes, trace, citations cùng review. Giao diện có ô hỏi đáp, hiển thị tuyến Agent, trạng thái Guard và link nguồn. Sáu chức năng dự đoán cũ vẫn dùng các checkpoint đã train.")
    code('''@app.post("/api/agent")
def agent_answer():
    question = request.get_json(silent=True).get("question", "").strip()
    if not question:
        return jsonify({"error": "Thiếu câu hỏi."}), 400
    return jsonify(AGENT.run(question))''')
    p("Render dùng requirements-web.txt, Gunicorn và health endpoint. Phần Knowledge/Agent chỉ phụ thuộc JSON và NumPy nên không làm image deploy phải cài TensorFlow hoặc PyTorch. Repository tieuluan3 chứa render.yaml để người dùng có thể tạo Blueprint từ GitHub.")

    page("6.5. Kết luận phần cập nhật")
    p("Chương 5 và 6 đã nối sáu bài toán dự đoán của quyển cũ với lớp tri thức và điều phối. Hệ thống mới dùng 12 tài liệu, 23 triples, benchmark 15 câu, năm phương pháp truy hồi và bảy agent. Hai notebook 07–08 giải thích code và đã lưu output.")
    p("Điểm chính của thực nghiệm là báo cáo đồng thời độ phủ và thứ hạng: B4 có Recall@3 cao nhất nhưng B0/B2 có MRR cao hơn. Knowledge Graph hữu ích để biểu diễn đường đi và hỗ trợ câu nhiều bước; nó không mặc nhiên thắng mọi metric. Multi-Agent cung cấp trace và ranh giới trách nhiệm, nhưng chưa thay thế đánh giá chuyên gia.")
    table(["Sản phẩm mới", "Vị trí"], [
        ["Notebook Knowledge Graph + RAG", "notebooks/07_knowledge_graph_rag.ipynb"],
        ["Notebook Multi-Agent", "notebooks/08_multi_agent_system.ipynb"],
        ["Corpus / graph / benchmark", "knowledge/*.json"],
        ["Kết quả truy hồi", "results/knowledge_rag"],
        ["Kết quả Agent", "results/multi_agent"],
        ["API", "POST /api/agent"],
    ], [210, 270])
