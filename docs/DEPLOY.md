# Deploy Render

Web live: https://tieuluan-vinh-intellilab.onrender.com (Free). Đã kiểm tra 18 tuyến dự đoán ngày 08/10/2026.

1. Đăng nhập https://dashboard.render.com/.
2. New → Blueprint → chọn https://github.com/Vinhdiesel28/tieuluan3, branch main.
3. Đọc `render.yaml`, kiểm tra service `tieuluan-vinh-intellilab`, plan **Free**.
4. Build: `pip install -r requirements-web.txt`.
5. Start: `gunicorn --chdir web app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2`.
6. Khi Live, `/health` phải trả `status: ok, models: 38`. Thử cả sáu bài, các kiến trúc và nhóm đối chiếu ba framework.

Không cần database hoặc token. Bundle đã có trọng số thật; server không train lại. File docs/deployment_status.json ghi trạng thái live đã xác minh, không suy ra online chỉ vì có YAML.

Chạy local: `python -m pip install -r requirements-web.txt`, sau đó `python web/app.py`; mở http://127.0.0.1:5007.

CSV giá nhà/CDC có một hàng và header đúng tên cột. Khách hàng tám hàng tuần từ cũ đến mới, tiền GBP. AAPL ít nhất80 phiên với Date,Open,High,Low,Close,Volume. Ảnh MNIST/EuroSAT được resize16×16 theo đúng train. CDC Age là mã nhóm tuổi, không phải tuổi tính bằng năm.
