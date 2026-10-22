# Tiểu luận 03 — Phát triển các hệ thống thông minh

**Sinh viên:** Nguyễn Minh Vinh · **MSV:** B23DCCN934 · **Lớp:** D23CTPM01

**Giảng viên:** PGS.TS Trần Đình Quế

Repository là quyển tiểu luận hoàn chỉnh từ lịch sử AI, ML, CNN, RNN đến Knowledge Graph, RAG và hệ đa Agent. Chương 5–6 là phần cập nhật mới và được trình bày bằng **màu đỏ** trong báo cáo.

## Sản phẩm nộp

- [Báo cáo PDF hoàn chỉnh](report/tieuluan03_CT_08_VinhNM.934.pdf) và [bản Markdown](report/Tieu_luan_03_Nguyen_Minh_Vinh.md).
- 8 notebook đã chạy và lưu output trong [`notebooks/`](notebooks).
- Mã nguồn chung trong [`src/`](src), kết quả thực nghiệm trong [`results/`](results).
- Web Flask gồm sáu chức năng dự đoán cũ và giao diện hỏi đáp Multi-Agent có trace, citation và Guard review.
- Blueprint Render tại [`render.yaml`](render.yaml).

| Chương | Nội dung | Dataset / tri thức | Notebook |
|---|---|---|---|
| 1 | Lịch sử phát triển AI và khảo sát dữ liệu | Tổng quan sáu dataset | `01_history_and_data.ipynb` |
| 2 | ML cơ bản, scratch/Keras/PyTorch | CDC Diabetes, giá nhà Việt Nam | `02_ML_experiments.ipynb` |
| 3 | CNN, scratch/Keras/PyTorch | MNIST, EuroSAT | `03_CNN_experiments.ipynb` |
| 4 | RNN, LSTM và so sánh kiến trúc | AAPL, Online Retail II | `04_RNN_experiments.ipynb`, `06_architecture_comparison.ipynb` |
| 5 | Knowledge Base, Knowledge Graph và RAG | Diabetes, AAPL, 12 tài liệu, 23 triples, 15 câu benchmark | `07_knowledge_graph_rag.ipynb` |
| 6 | Hệ đa Agent | Kết quả sáu bài toán và kho tri thức | `08_multi_agent_system.ipynb` |

Notebook `05_comparison_deploy.ipynb` tổng hợp mô hình và triển khai. Notebook 07 có 20 code cell đã chạy, bao phủ validation, ingestion/chunking, provenance, B0–B4, grounded answer, request contract, metric, ablation, phân tích lỗi và API. Notebook 08 giải thích hệ đa Agent.

## Dataset

Hai tệp được đưa trực tiếp lên GitHub:

- `data/diabetes.csv`: CDC BRFSS 2015, 253.680 dòng × 22 cột; bài ML gộp nhãn tiền tiểu đường và tiểu đường thành lớp dương.
- `data/AAPL_2015_2025.csv`: 2.766 phiên × 7 cột, từ 02/01/2015 đến 31/12/2025; dùng chuỗi OHLCV dự báo Close phiên kế tiếp.

Các bộ lớn MNIST, EuroSAT, Online Retail II và giá nhà không được nhân bản trong Git. URL, kích thước và SHA-256 nằm tại [`docs/data_manifest.json`](docs/data_manifest.json); `tools/download_data.py` tải các nguồn công khai. Corpus của Chương 5 nằm tại [`knowledge/`](knowledge) và có URL nguồn trong từng tài liệu.

## Kết quả Chương 5–6

| System | Method | Precision@3 | Recall@3 | MRR@3 | nDCG@3 | OOD abstain |
|---|---|---:|---:|---:|---:|---:|
| B0 | BM25 | 0,4167 | 0,9167 | **0,9583** | **0,9175** | 1,0000 |
| B1 | LSA | 0,3889 | 0,8750 | 0,9167 | 0,8493 | 1,0000 |
| B2 | Hybrid | 0,4167 | 0,9167 | **0,9583** | 0,9056 | 1,0000 |
| B3 | KG | 0,3611 | 0,7083 | 0,5972 | 0,6112 | 1,0000 |
| B4 | GraphRAG | **0,4444** | **0,9375** | 0,9167 | 0,9063 | 1,0000 |

Hệ thống có 7 Agent và 6 kịch bản: 3 `supported`, 3 `needs_human`. Đây là trạng thái policy, không phải độ chính xác mô hình.

## Chạy notebook và web

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

Chạy web nhẹ, không cần TensorFlow/PyTorch:

```bash
python -m pip install -r requirements-web.txt
python web/app.py
```

Mở `http://127.0.0.1:5007`. API Agent:

```http
POST /api/agent
Content-Type: application/json

{"question":"AAPL được dự báo bằng dữ liệu nào và có nên mua không?"}
```

API RAG độc lập hỗ trợ năm cấu hình B0–B4:

```http
POST /api/rag
Content-Type: application/json

{"question":"AAPL gồm dữ liệu nào?","method":"B4","top_k":3,"hops":2,"domain":"finance"}
```

Tạo lại phần mới và báo cáo:

```bash
python tools/run_agent_experiments.py
python tools/build_agent_notebooks.py
python tools/execute_agent_notebooks.py
python tools/build_report.py
```

Các số liệu trong báo cáo được đọc từ file kết quả. B4 có Recall@3 cao nhất nhưng B0/B2 có MRR cao hơn, vì vậy không tuyên bố một hệ thống thắng mọi tiêu chí. Kết quả sức khỏe và tài chính chỉ phục vụ học tập, không thay thế chẩn đoán hoặc khuyến nghị đầu tư.
