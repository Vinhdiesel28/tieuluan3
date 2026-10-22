# Đối chiếu và bổ sung Chương 5

Nguồn đối chiếu:

- Báo cáo mẫu `tieuluan03_CT_08_anhtt.pdf` do người dùng cung cấp.
- Repository tham khảo: <https://github.com/tuananhtrieu1305/ISD_Assignment01>.

Không sao chép kết quả hoặc đổi sang MedQuAD/FinQA/Wikidata của bài mẫu. Tiểu luận này giữ CDC Diabetes, AAPL và corpus có nguồn đã dùng từ phiên bản trước.

| Thành phần Chương 5 | Trước đối chiếu | Sau bổ sung |
|---|---|---|
| Corpus và graph | 12 documents, 23 triples | Giữ nguyên, bổ sung validation và hash snapshot |
| Ingestion/chunking | Chưa có artifact chunk | 23 chunk, cửa sổ 45 token, overlap 8, URL và SHA-256 |
| Schema/provenance | Evidence ID cơ bản | Kiểm tra required fields, ID trùng, HTTPS URL, dangling evidence và gold ID |
| Hệ so sánh | Tên BM25/LSA/KG/Hybrid/GraphRAG | Khóa contract B0–B4 giống logic bài mẫu |
| Answer layer | Ghép đoạn retrieved | Chọn câu grounded, citation, reason và abstention |
| Input contract | Kiểm tra câu hỏi ở Agent | JSON Schema và `validate_request` cho method/top-k/hops/domain |
| Metric | Recall, MRR, Complete | Thêm Precision, nDCG, OOD abstention, p50/p95 latency |
| Ablation | Chưa có | Graph hops 0–3 × top-k 1/3/5 |
| Prediction artifact | Kết quả tổng hợp | 75 prediction rows có score, citation, latency và hash |
| Error analysis | Nhận xét bằng văn bản | CSV theo query, system và gold bị thiếu |
| API | `/api/agent` | Thêm `/api/rag` cho B0–B4 |
| Notebook 07 | 7 code cell | 20 code cell đã chạy, không có error output |
| Báo cáo | 8 trang Chương 5 | 11 trang Chương 5, mục 5.1–5.10 màu đỏ |

Phần không đưa vào:

- Không tuyên bố đây là Microsoft GraphRAG đầy đủ hoặc RAG có LLM sinh văn bản.
- Không dựng Neo4j vì adjacency/BFS cục bộ đã đủ cho bài và deploy Render nhẹ.
- Không dùng số liệu, checkpoint hoặc kết quả MedQuAD/FinQA của người khác.
- Không tạo ba implementation Keras/PyTorch cho retrieval vì yêu cầu ba framework đã được đáp ứng ở ML/CNN/RNN; làm như vậy cho BM25/KG sẽ không phải đối chiếu tương đương.
