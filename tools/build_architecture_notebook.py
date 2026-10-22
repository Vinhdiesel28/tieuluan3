"""Add the architecture-comparison notebook without resetting existing outputs."""
from pathlib import Path
import nbformat as n
ROOT=Path(__file__).resolve().parents[1]
md=n.v4.new_markdown_cell;code=n.v4.new_code_cell
cells=[md('''# 06 — So sánh các mô hình/kiến trúc khác nhau
Nguyễn Minh Vinh · B23DCCN934.

Notebook bổ sung theo hướng bài Phạm Văn Tư. Dataset giữ nguyên: CDC và giá nhà; MNIST và EuroSAT; AAPL và Online Retail II.

- ML: Logistic Regression/Ridge, Random Forest, MLP.
- CNN: CNN cơ bản, VGG-style, MobileNet-style, ResNet-style.
- RNN: Simple RNN, LSTM, GRU, BiLSTM.

Notebook 02–04 đã có scratch/Keras/PyTorch cho kiến trúc cơ sở. Notebook này so sánh **kiến trúc khác nhau**, CNN/RNN được train từ đầu bằng PyTorch. Các bản “style” là mạng nhỏ 16×16, không phải VGG-11/MobileNetV2/ResNet-18 đầy đủ hoặc pretrained.

Chạy tuần tự từ trên xuống. `RETRAIN=False` đọc checkpoint đã train và tạo lại bảng/hình; đổi thành `True` để train lại toàn bộ. Mỗi lần huấn luyện mới sẽ ghi đè kết quả kiến trúc tương ứng.'''),code('''import os,sys,json
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/architectures.py').exists())
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from IPython.display import display,Image,Code
import pandas as pd
RETRAIN=False'''),md('''## 1. Các kiến trúc CNN và RNN
CNN dùng tensor NHWC đầu vào, chuyển NCHW khi tích chập. VGG-style tăng độ sâu bằng kernel 3×3; MobileNet-style tách depthwise và pointwise, có nhánh mở rộng/chiếu tuyến tính; ResNet-style cộng nhánh tắt.

RNN/LSTM/GRU dùng hidden size 32. BiLSTM nối trạng thái cuối hai chiều của **cửa sổ lịch sử đã quan sát**, không đọc ngày/tuần chứa target. BiLSTM nhiều tham số hơn nên phép so sánh không phải cùng số tham số.'''),code((ROOT/'src/architectures.py').read_text(encoding='utf-8')),md('''## 2. Quy trình huấn luyện có kiểm soát
Cùng split và thống kê train của notebook trước; không tạo lại tập test. Bốn kiến trúc trong cùng bài dùng Adam lr=0,001, batch=128, 20 epoch, seed=42 và clip norm=1. Class weights chỉ tính từ train. Checkpoint chọn bằng validation loss, kiến trúc chọn bằng validation Macro-F1/RMSE. Không chọn theo test.

Không diễn giải khác biệt giữa nhóm Adam mới và nhóm SGD cũ chỉ là tác dụng của kiến trúc. Thời gian CPU phụ thuộc máy; không bịa FLOPs/FPS. Một seed chưa cho phép kết luận thống kê.''')]
source=(ROOT/'tools/train_architectures.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
source='\n'.join(line for line in source.splitlines() if not line.startswith(('ROOT=','from architectures import')))
cells.append(code(source))
cells += [md('''## 3. ML: ba họ thuật toán trên cùng dữ liệu
Đọc kết quả Logistic Regression/Ridge và Random Forest đã train cùng split với MLP. MLP đại diện dùng checkpoint PyTorch của phép đối chiếu thư viện. Đây là ba thuật toán khác nhau; không tính ba framework thành ba thuật toán. Code fit và xuất checkpoint bên dưới.'''),code("display(Code(filename=str(ROOT/'tools/baselines.py'),language='python'))"),code('''import runpy
if RETRAIN or not (ROOT/'results/diabetes/classical/RandomForest.npz').exists():
    runpy.run_path(str(ROOT/'tools/baselines.py'),run_name='__main__')
for key in ['diabetes','housing']:
    classical=pd.read_csv(ROOT/'results'/key/'classical_baselines.csv')
    mlp=json.loads((ROOT/'results'/key/'pytorch_42.json').read_text())
    result=pd.concat([classical,pd.DataFrame([dict(Model='MLP',**mlp['test'])])],ignore_index=True)
    print(key);display(result.round(5))
    result.to_csv(ROOT/'results'/key/'model_comparison.csv',index=False)''')]
for key,names in [('mnist',['basic_cnn','vgg_style','mobilenet_style','resnet_style']),('eurosat',['basic_cnn','vgg_style','mobilenet_style','resnet_style']),('stock',['simple_rnn','lstm','gru','bilstm']),('customer',['simple_rnn','lstm','gru','bilstm'])]:
    cells += [md(f'## Dataset {key}\nĐầu vào và target lấy nguyên cache đã chuẩn hóa. Hiển thị quy mô và protocol trước khi đánh giá.'),code(f"data=load('{key}')\ndisplay(pd.Series(data['meta']))\nprint('Train/val/test:',*[len(data[s]) for s in ['train','val','test']])")]
    for name in names:
        cells += [md(f'### {name}\nNếu RETRAIN=True hoặc thiếu checkpoint, cell huấn luyện từ đầu. Nếu checkpoint đã có, đọc kết quả thực nghiệm đã lưu.'),code(f"path=ROOT/'results/architectures/{key}/{name}.json'\nrow=run('{key}','{name}') if RETRAIN or not path.exists() else json.loads(path.read_text(encoding='utf-8'))\ndisplay(pd.Series(row))")]
    cells += [md('### Bảng so sánh, learning curve và lỗi\nĐánh giá cùng test; mặc định chọn theo validation. Kiểm tra baseline, đặc biệt AAPL với giá phiên trước.'),code(f"display(summarize('{key}').round(5))\nfor file in ['learning.png','comparison.png']:\n    display(Image(filename=str(ROOT/'results/architectures/{key}'/file)))\nprint((ROOT/'results/architectures/{key}/selection.json').read_text())\nprint('Baseline:',(ROOT/'results/{key}/baseline.json').read_text())\ndisplay(pd.read_json(ROOT/'results/architectures/{key}/errors.json'))")]
cells += [md('''## 4. Đóng gói và kiểm chứng
Web chọn từng kiến trúc thực đã train. NumPy phục vụ trọng số export từ PyTorch, bao gồm đúng thứ tự gate của LSTM/GRU và trạng thái ngược BiLSTM. Đây là **inference NumPy**, không tự nhận đã viết backward scratch cho mọi kiến trúc mới. Scratch training đầy đủ của kiến trúc cơ sở nằm trong notebook 02–04.

Chạy `tools/verify_architectures.py` để nạp lại checkpoint native, so toàn bộ dự đoán test với bản NumPy và kiểm tra API.'''),code("display(Code(filename=str(ROOT/'src/architecture_inference.py'),language='python'))"),code("runpy.run_path(str(ROOT/'tools/bundle_architectures.py'),run_name='__main__')"),md('''## 5. Nhận xét và giới hạn
Không thay dataset bằng dữ liệu bài Tư và không sao chép số đo của bài mẫu. So sánh 4 kiến trúc CNN/RNN và 3 họ mô hình ML đáp ứng phần đánh giá mô hình khác nhau; phần ba framework là một phép kiểm tra riêng.

CNN ảnh 16×16, mô hình nhỏ, chưa augmentation/pretrained/spatial split; RNN chỉ một seed và một split thời gian. BiLSTM chỉ nhìn ngược trong cửa sổ quá khứ. Đường học và baseline giúp thấy mô hình lớn hơn có thể không tốt hơn.''')]
book=n.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
n.write(book,ROOT/'notebooks/06_architecture_comparison.ipynb')
print('Created architecture comparison notebook')
