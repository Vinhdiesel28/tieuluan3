# Dữ liệu được nộp cùng Tiểu luận 03

Hai tệp CSV dưới đây được giữ trực tiếp trong repository để notebook Chương 5 và hệ thống Agent có thể chạy ngay. Các bộ dữ liệu lớn còn lại của Chương 2–4 được tải bằng `tools/download_data.py`; URL, kích thước và SHA-256 được ghi trong `docs/data_manifest.json`.

| Tệp | Nguồn | Quy mô | Vai trò |
|---|---|---:|---|
| `diabetes.csv` | CDC BRFSS 2015, bản Health Indicators do Alex Teboul công bố lại | 253.680 dòng, 22 cột gồm nhãn ba mức | Dữ liệu bảng sức khỏe; bài thực nghiệm gộp tiền tiểu đường và tiểu đường thành lớp dương |
| `AAPL_2015_2025.csv` | Dữ liệu AAPL trong repository tham khảo A06 | 2.766 phiên, 7 cột, từ 02/01/2015 đến 31/12/2025 | Chuỗi OHLCV cho dự báo Close phiên kế tiếp và miền tri thức tài chính |

Nguồn gốc không bị thay đổi trong Tiểu luận 03. Kho `knowledge/corpus.json` chỉ chứa 12 đoạn văn có dẫn nguồn để đánh giá truy hồi, không thay thế hai CSV và không được dùng làm dữ liệu huấn luyện MLP/RNN.
