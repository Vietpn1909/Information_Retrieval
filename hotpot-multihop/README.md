# Truy xuất đa bước trên HotpotQA

**Từ chỉ mục ngược đến truy xuất lặp có dẫn hướng** — Bài tập lớn môn Truy vấn dữ liệu (IT5411), HUST.

## Bài toán

Cho một câu hỏi và một kho khoảng 106 nghìn đoạn văn Wikipedia, hệ thống phải **tìm ra những đoạn
văn chứa đủ thông tin để trả lời**. Điểm khó: nhiều câu hỏi cần ghép thông tin từ **hai đoạn khác
nhau**, và đoạn thứ hai thường không có từ nào trùng với câu hỏi.

Ví dụ (lấy từ dữ liệu thật):

> *What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?*

1. **Bước 1:** tìm đoạn về phim *Kiss and Tell (1945)*. Đoạn này cho biết vai Corliss Archer do **Shirley Temple** đóng.
2. **Bước 2:** tìm đoạn về *Shirley Temple*. Đoạn này cho biết bà từng là **Chief of Protocol**, chính là câu trả lời.

Câu hỏi không hề nhắc đến "Shirley Temple", nên tìm kiếm thông thường một lượt rất khó lấy được
đoạn thứ hai. Đo trên dữ liệu: với **gần 20% câu hỏi loại này, đoạn thứ hai không lọt nổi vào
top-100** kết quả của BM25. Đó là lý do cần **truy xuất đa bước**: dùng kết quả bước 1 để tìm ra
"thực thể cầu nối" (Shirley Temple), rồi tạo truy vấn mới cho bước 2.

## Hướng tiếp cận

```
Câu hỏi ──► Truy xuất bước 1 ──► Trích thực thể cầu nối ──► Truy xuất bước 2 ──► Gộp & xếp hạng ──► Danh sách đoạn văn
            (BM25 / dense / lai)   (vd. "Shirley Temple")      (truy vấn mới)                         (→ câu trả lời)
```

Trọng tâm của dự án là **tầng truy xuất**: đầu ra chính là danh sách đoạn văn đã xếp hạng, còn
việc sinh câu trả lời chỉ là bước cuối. Theo đúng nội dung môn học, chỉ mục ngược, TF-IDF và BM25
được **tự cài đặt từ đầu**, không dùng thư viện tìm kiếm có sẵn. Mọi phương pháp xuất kết quả cùng
một định dạng (TREC run) và được chấm bằng cùng một bộ độ đo (Recall, nDCG, MAP, MRR), nên so sánh
trực tiếp được với nhau.

## Dữ liệu

[HotpotQA](https://hotpotqa.github.io/) (bản *distractor*): 5.000 câu hỏi lấy mẫu từ tập train
(dùng để tinh chỉnh tham số) và 7.405 câu của tập dev (dùng để báo cáo kết quả). Ngữ cảnh của mọi
câu hỏi được gộp thành **một kho chung 105.862 đoạn văn**, nên hệ thống phải tìm trong toàn bộ kho
chứ không chỉ trong 10 đoạn đi kèm mỗi câu.

## Phân công và tiến độ

Khung dự án dùng chung đã sẵn sàng: pipeline dữ liệu, chỉ mục ngược cơ bản, harness chấm điểm và
interface giữa các module. Các thành viên phát triển phần của mình trên khung này.

| Thành viên | Phần việc | Tiến độ |
|---|---|---|
| Nguyên | Chỉ mục ngược, chuẩn hoá văn bản, nén postings (delta + VByte) | Chưa bắt đầu |
| Việt | Truy xuất thưa: TF-IDF, BM25, tinh chỉnh k1/b, skip pointer, champion list, chỉ mục phân tầng | ✅ Xong |
| Khiêm | Truy xuất dày & lai: BGE-small, FAISS, RRF, rerank bằng cross-encoder | Chưa bắt đầu |
| Cường | Truy xuất đa bước: sinh truy vấn bước 2, thực thể cầu nối, điều kiện dừng | Chưa bắt đầu |
| Vinh | Đánh giá & demo: harness chấm điểm, ablation, đồ thị, phân tích lỗi, giao diện | Chưa bắt đầu |

## Kết quả hiện có (tập dev)

| Phương pháp | Recall@100 | nDCG@10 | Thời gian / câu |
|---|---|---|---|
| TF-IDF (lnc.ltc) | 0.909 | 0.683 | ~20 ms |
| BM25 đã tinh chỉnh | **0.910** | **0.721** | ~19 ms |
| BM25 + champion list (r = 500) | 0.905 | 0.721 | **~8 ms** |

Các con số này là **mốc cơ sở**: phần truy xuất dày và đa bước sẽ được so sánh với chúng.

## Chạy thử

Cần [uv](https://docs.astral.sh/uv/) và Python 3.11.

```bash
uv sync                                                                   # cài môi trường
uv run python scripts/build_corpus.py                                     # tải dữ liệu, dựng kho đoạn văn
uv run python scripts/build_index.py --index stem                         # dựng chỉ mục ngược
uv run python scripts/run_retrieval.py --config bm25_tuned --split dev --k 100   # chạy truy xuất
uv run python scripts/evaluate.py --run runs/bm25_tuned.txt               # chấm điểm
```

## Cấu trúc thư mục

```
config/      tham số: default.yaml (dùng chung) + parts/<phần>.yaml (riêng từng thành viên)
src/hmr/     mã nguồn: data/ (dữ liệu), index/ (chỉ mục), retrieval/ (truy xuất),
             multihop/ (đa bước), eval/ (đánh giá), common/ (interface dùng chung)
scripts/     các lệnh chạy: dựng dữ liệu, dựng chỉ mục, truy xuất, chấm điểm, thí nghiệm
tests/       kiểm thử tự động (uv run pytest)
docs/        tài liệu chi tiết
```

## Tìm hiểu thêm

- **Thành viên nhóm — đọc trước khi code:** [docs/HUONG_DAN_THANH_VIEN.md](docs/HUONG_DAN_THANH_VIEN.md)
  (đặc tả dữ liệu, những gì đã có, ai sở hữu file nào, cách thêm phần mới mà không conflict).
- **Chi tiết cài đặt, toàn bộ bảng số liệu và minh chứng kiểm thử:** [docs/CHI_TIET.md](docs/CHI_TIET.md).
