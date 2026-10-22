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
    agents = json.loads((root / "results/multi_agent/summary.json").read_text(encoding="utf-8"))

    page("CHƯƠNG 5. KNOWLEDGE GRAPH VÀ RAG")
    p("PHẦN CẬP NHẬT MỚI CỦA TIỂU LUẬN 03.")
    p("Chương này mở rộng quyển tiểu luận từ mô hình dự đoán sang hệ thống trả lời có bằng chứng. Mô hình ML, CNN hoặc RNN ánh xạ đầu vào thành nhãn hay giá trị; RAG nhận một câu hỏi, truy hồi các đoạn liên quan rồi tạo câu trả lời kèm nguồn. Knowledge Graph bổ sung các thực thể và quan hệ để truy vấn không chỉ phụ thuộc vào từ khóa.")
    p("Thiết kế được biểu diễn như hợp hàm: answer = compose(review, generate, retrieve, normalize)(question). Mỗi hàm có đầu vào, đầu ra và trách nhiệm riêng. Nhờ đó có thể thay BM25 bằng LSA, thêm mở rộng đồ thị hoặc đổi cách tổng hợp mà không làm thay đổi toàn bộ hệ thống.")
    table(["Thành phần", "Đầu vào", "Đầu ra", "Mục đích"], [
        ["Normalize", "Câu hỏi Unicode", "Token chuẩn hóa", "Giảm khác biệt chữ hoa và dấu"],
        ["Retriever", "Token và chỉ mục", "Danh sách tài liệu", "Tìm bằng chứng phù hợp"],
        ["Knowledge Graph", "Thực thể hạt giống", "Đường đi và evidence", "Mở rộng quan hệ nhiều bước"],
        ["Generator", "Câu hỏi và bằng chứng", "Câu trả lời có trích dẫn", "Không trả lời ngoài dữ liệu"],
    ], [100, 115, 120, 145])

    page("5.1. Cơ sở Knowledge Graph")
    p("Knowledge Graph biểu diễn tri thức bằng bộ ba (subject, predicate, object). Subject và object là thực thể; predicate mô tả quan hệ có hướng. Mỗi cạnh trong bài còn mang trường evidence trỏ về một tài liệu trong corpus. Ràng buộc này cho phép kiểm tra rằng quan hệ dùng để trả lời có nguồn, thay vì tạo một cạnh không thể truy vết.")
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

    page("5.2. Truy hồi BM25 và LSA")
    p("BM25 là phương pháp truy hồi từ vựng. Điểm tăng khi từ của câu hỏi xuất hiện nhiều trong tài liệu, nhưng được điều chỉnh theo độ dài tài liệu và độ hiếm IDF. Bài dùng k1=1,5 và b=0,75. Ưu điểm là nhanh, dễ giải thích và phù hợp corpus nhỏ; nhược điểm là khó nối các cách diễn đạt khác nhau.")
    p("LSA xây ma trận TF-IDF tài liệu–từ, chuẩn hóa từng hàng và phân rã SVD. Câu hỏi được chiếu vào không gian tám chiều rồi xếp hạng theo cosine similarity. Đây là mô hình dense cổ điển chạy hoàn toàn offline bằng NumPy. Nó không được mô tả như embedding ngôn ngữ hiện đại và không dùng dịch vụ bên ngoài.")
    code('''tfidf = normalize_rows(term_frequency * inverse_document_frequency)
_, _, vt = np.linalg.svd(tfidf, full_matrices=False)
projection = vt[:8].T
dense_docs = tfidf @ projection
score = cosine(dense_docs, query_tfidf @ projection)''')
    p("Hai danh sách BM25 và LSA được kết hợp bằng Reciprocal Rank Fusion: RRF(d)=Σ w/(60+rank). Phương pháp dùng thứ hạng thay cho thang điểm gốc nên tránh phải ép điểm BM25, cosine và đồ thị về cùng đơn vị.")

    page("5.3. GraphRAG và hợp hàm")
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

    page("5.4. Dataset của Chương 5")
    p("Hai CSV được chọn từ chính các chương cũ để quyển có tính liên tục. Diabetes cung cấp bối cảnh sức khỏe cho MLP và kho tri thức; AAPL cung cấp chuỗi OHLCV cho RNN và miền tài chính. Corpus không chứa thông tin cá nhân, không lấy mẫu dự đoán làm chẩn đoán hay khuyến nghị đầu tư.")
    table(["Dataset", "Kích cỡ", "Output / nhãn", "Nguồn và vai trò"], [
        ["CDC Diabetes", "253.680 × 22", "Diabetes_012; bài gộp 1/2 thành dương", "BRFSS 2015; dữ liệu bảng sức khỏe"],
        ["AAPL 2015–2025", "2.766 × 7", "Close phiên kế tiếp", "OHLCV; chuỗi tài chính"],
        ["Knowledge corpus", f"{len(corpus)} tài liệu", "Đoạn bằng chứng và URL", "WHO, CDC, NIDDK, Apple và metadata bài"],
        ["Knowledge graph", f"{len(graph)} triples", "subject–predicate–object", "Evidence liên kết corpus"],
        ["Retrieval benchmark", f"{len(benchmark)} câu", "Gold document ids", "Direct, paraphrase, multi-hop, OOD"],
    ], [105, 95, 145, 135])
    p("Mỗi tài liệu ghi id, domain, title, url, as_of và text. Benchmark được tách theo loại câu hỏi để tránh đánh giá chỉ bằng các câu trùng nguyên văn. Câu out-of-scope kiểm tra khả năng không đưa bằng chứng sai; đây là một phần của chất lượng hệ thống chứ không phải lỗi cần che giấu.")

    page("5.5. Mã nguồn và cách chạy notebook")
    p("Notebook 07 giải thích lần lượt khâu nạp corpus, dựng BM25/LSA, dựng graph, GraphRAG, truy vấn minh họa và benchmark. Các cell đã có output, có thể Restart Kernel rồi Run All. Mã nghiệp vụ được đặt trong src/knowledge_rag.py để notebook, web và script đánh giá dùng cùng một implementation.")
    code('''from src.knowledge_rag import KnowledgeRAG

rag = KnowledgeRAG()
result = rag.answer(
    "AAPL được dự báo bằng dữ liệu nào và có phải khuyến nghị đầu tư không?",
    method="GraphRAG", top_k=3, hops=2,
)
for source in result["citations"]:
    print(source["title"], source["url"])''')
    p("Việc giữ nguồn trong cấu trúc answer giúp giao diện hiển thị link thật thay vì chỉ in một đoạn văn. Cùng câu hỏi và cùng corpus cho kết quả xác định, thuận lợi cho chấm bài và phân tích lỗi.")

    page("5.6. Kết quả đánh giá truy hồi")
    rows = []
    for row in scores:
        rows.append([row["Method"], f'{float(row["Recall@3"]):.4f}', f'{float(row["MRR@3"]):.4f}', f'{float(row["Complete@3"]):.4f}', f'{float(row["OOD_EmptyRate"]):.4f}', f'{float(row["Latency_ms"]):.3f}'])
    table(["Method", "Recall@3", "MRR@3", "Complete@3", "OOD empty", "ms"], rows, [80, 75, 70, 85, 85, 65])
    fig(root / "results/knowledge_rag/retrieval_comparison.png", "So sánh năm phương pháp trên benchmark 15 câu.", maxheight=260)
    p("Hybrid đạt Recall@3 cao nhất 0,9375. GraphRAG đạt 0,9167, bằng BM25 về Recall và không vượt Hybrid trong corpus nhỏ này. Kết quả được giữ nguyên vì mục tiêu là báo cáo đúng phép đo, không ép phương pháp mới phải thắng. KG có OOD empty rate 0,6667, tốt hơn các retriever luôn tìm thấy một đoạn, nhưng Recall thấp hơn.")

    page("5.7. Nhận xét, sai số và giới hạn")
    p("Sai số thứ nhất đến từ alias: graph chỉ khởi động khi câu hỏi chứa tên thực thể đã biết. Sai số thứ hai đến từ corpus nhỏ, khiến SVD có ít ngữ cảnh và một tài liệu có thể chứa nhiều khái niệm. Sai số thứ ba là phép ghép bằng chứng: RRF cải thiện độ phủ nhưng tài liệu được thêm từ graph có thể làm mất một tài liệu từ vựng ở top-3.")
    p("Hệ thống hiện là extractive RAG: câu trả lời ghép nội dung nguồn, chưa có mô hình sinh. Vì vậy nó ít hallucination nhưng diễn đạt kém tự nhiên hơn LLM. Nếu thay bằng mô hình sinh cần thêm prompt grounding, kiểm tra citation entailment, chống prompt injection và đánh giá factuality.")
    table(["Rủi ro", "Biện pháp hiện tại", "Hướng tiếp theo"], [
        ["Trả lời ngoài phạm vi", "Thông báo không có bằng chứng", "Ngưỡng score được hiệu chỉnh"],
        ["Quan hệ sai", "Mỗi triple bắt buộc evidence id", "Kiểm duyệt graph và versioning"],
        ["Nguồn lỗi thời", "Có as_of và URL", "Lịch làm mới và snapshot"],
        ["Diễn giải quá mức", "Guard notice sức khỏe/tài chính", "Human review theo rủi ro"],
    ], [105, 175, 200])

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
    p("Điểm chính của thực nghiệm là tính trung thực: Hybrid có Recall@3 cao nhất, GraphRAG không được tuyên bố vượt trội khi số liệu không ủng hộ. Knowledge Graph vẫn hữu ích để biểu diễn đường đi và hỗ trợ từ chối ngoài phạm vi. Multi-Agent cung cấp trace và ranh giới trách nhiệm, nhưng chưa thay thế đánh giá chuyên gia.")
    table(["Sản phẩm mới", "Vị trí"], [
        ["Notebook Knowledge Graph + RAG", "notebooks/07_knowledge_graph_rag.ipynb"],
        ["Notebook Multi-Agent", "notebooks/08_multi_agent_system.ipynb"],
        ["Corpus / graph / benchmark", "knowledge/*.json"],
        ["Kết quả truy hồi", "results/knowledge_rag"],
        ["Kết quả Agent", "results/multi_agent"],
        ["API", "POST /api/agent"],
    ], [210, 270])
