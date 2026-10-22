"""Additional report sections, called in the builder's paragraph/table context."""
def architecture_sections(kind, page, p, table, fig, code, ROOT, read, js, f):
    cnn=kind=='cnn';prefix='3' if cnn else '4';keys=['mnist','eurosat'] if cnn else ['stock','customer']
    page(prefix+'.12. So sánh các kiến trúc khác nhau')
    if cnn:
        p('Phần bổ sung theo hướng bài mẫu Phạm Văn Tư so sánh CNN cơ bản, VGG-style, MobileNet-style và ResNet-style. Đây là bốn kiến trúc khác nhau được train độc lập bằng PyTorch, tách khỏi phép kiểm tra ba thư viện ở các phần trước. Dữ liệu, split, chuẩn hóa và ảnh 16×16 được giữ nguyên; không lấy kết quả hoặc trọng số của bài mẫu.')
        table(['Kiến trúc','Cấu trúc và khác biệt'],[
            ['CNN cơ bản','Conv 3×3, 8 kênh → ReLU → AvgPool → Dense'],
            ['VGG-style','Bốn Conv 3×3, kênh 16/32, hai lần pooling, Dense 64'],
            ['MobileNet-style','Mở rộng 1×1 → depthwise 3×3 → chiếu tuyến tính 1×1; một skip connection'],
            ['ResNet-style','Stem 16 kênh, hai residual block 16/32 kênh, global average pooling']], [110,370])
        p('VGG dùng chuỗi kernel nhỏ để tăng độ sâu [18]. MobileNetV2 dùng inverted residual và linear bottleneck [19]; bản nhỏ ở đây dùng ReLU, không BatchNorm, nên chỉ gọi MobileNet-style. ResNet học nhánh F(x) cộng với đường tắt x [20]; bản hai block này không phải ResNet-18 đầy đủ. Cả bốn đều khởi tạo ngẫu nhiên, không dùng pretrained.')
        code("# Residual block\nresidual = x\nx = torch.relu(conv1(x))\nx = conv2(x) + residual\nx = torch.relu(x)\n# Depthwise: mỗi kênh có kernel riêng\ndepth = nn.Conv2d(32, 32, 3, padding=1, groups=32)")
    else:
        p('Bốn kiến trúc được so sánh là Simple RNN, LSTM, GRU và BiLSTM. Dữ liệu vẫn là AAPL và Online Retail II, khác với cổ phiếu Việt Nam và bộ khách hàng mô phỏng trong bài Tư. Mỗi mô hình có một tầng hồi tiếp, hidden size 32; BiLSTM có 32 trạng thái mỗi chiều nên nhiều tham số hơn. Head trả log-return hoặc hai logits mua hàng.')
        table(['Kiến trúc','Cơ chế'],[['Simple RNN','h mới = tanh(Wx + Uh + b)'],['LSTM','Ba cổng input/forget/output, trạng thái cell và hidden'],['GRU','Cổng reset/update, một trạng thái hidden'],['BiLSTM','Hai LSTM đọc xuôi/ngược trong cửa sổ lịch sử; nối hai hidden cuối']], [110,370])
        p('BiLSTM chỉ đọc hai chiều bên trong 30 phiên hoặc 8 tuần đã quan sát trước thời điểm dự báo. Nó không được đọc phiên/tuần chứa nhãn. Vì vậy hai chiều không tự tạo leakage trong bài many-to-one này. Dùng h_n của cả hai chiều, không lấy output tại bước cuối một cách thiếu chiều ngược.')
        code("layer = nn.LSTM(input_size=D, hidden_size=32,\n                batch_first=True, bidirectional=True)\n_, (h, cell) = layer(x)\nfeatures = torch.cat((h[-2], h[-1]), dim=1)\nlogits = head(features)")
    p('Cùng seed 42, Adam learning rate 0,001, batch 128, 20 epoch và gradient clipping 1. Checkpoint lấy validation loss thấp nhất; kiến trúc mặc định chọn bằng validation Macro-F1 hoặc RMSE. Test chỉ dùng báo cáo sau khi cố định cấu hình. Không so trực tiếp mức tăng với nhóm SGD cũ để kết luận riêng tác dụng của kiến trúc, vì optimizer khác nhau. Toàn bộ code và learning curve nằm trong notebook 06.')
    page(prefix+'.13. Kết quả so sánh kiến trúc trên hai dataset')
    for key in keys:
        rows=read(ROOT/'results/architectures'/key/'comparison.csv')
        regression=key=='stock';columns=['RMSE','MAE','R2'] if regression else ['Accuracy','MacroF1']
        p('Dataset: '+{'mnist':'MNIST','eurosat':'EuroSAT','stock':'AAPL (USD)','customer':'Online Retail II'}[key])
        table(['Mô hình',*columns,'Tham số'],[[r['Model'],*[f(r[c]) for c in columns],r['Parameters']] for r in rows])
        choice=js(ROOT/'results/architectures'/key/'selection.json')
        chosen=next(r for r in rows if r['Model']==choice['name'])
        p('Chọn từ validation: '+choice['name']+'. Test '+choice['criterion']+' = '+f(chosen[choice['criterion']])+'.')
        if regression:
            base=js(ROOT/'results/stock/baseline.json')
            relation='thấp hơn' if float(chosen['RMSE'])<base['RMSE'] else 'cao hơn'
            p('Baseline giữ giá phiên trước có RMSE '+f(base['RMSE'])+' USD. Mô hình chọn trên validation có RMSE '+relation+' baseline; R² cao không thay thế đối chiếu này.')
    p('Các kiến trúc cùng dùng một tập chia nhưng khác số tham số và thời gian huấn luyện; chi phí chi tiết được lưu trong CSV/notebook. Đây là so sánh cùng ngân sách epoch, không phải cùng FLOPs. Kết quả một seed chưa đủ kết luận ưu thế thống kê. Mô hình phức tạp hơn không mặc định tốt hơn trên mọi bộ dữ liệu.')
    fig(ROOT/'results/architectures'/keys[1]/'comparison.png','So sánh test trên dataset thứ hai; mô hình được chọn bằng validation.',maxheight=150)
