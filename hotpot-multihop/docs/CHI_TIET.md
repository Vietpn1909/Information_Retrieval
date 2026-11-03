# Tài liệu kỹ thuật chi tiết — hotpot-multihop

> Tổng quan dự án xem [README](../README.md). File này ghi chi tiết cài đặt, quy ước, số liệu và minh chứng kiểm thử.

**Truy xuất đa bước trên HotpotQA — từ chỉ mục ngược đến truy xuất lặp có dẫn hướng**
Bài tập lớn IT5411 Truy vấn dữ liệu — HUST. Nhóm 5 người.

Trọng tâm là **tầng truy xuất**: output chính là danh sách đoạn văn đã xếp hạng
(`RetrievalResult`), câu trả lời chỉ là bước cuối. Chỉ mục ngược, tf-idf (lnc.ltc) và
BM25 được **tự cài đặt**, không dùng rank_bm25 / pyserini / elasticsearch.

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/) và Python 3.11.

```bash
uv sync                     # môi trường lõi + pytest
uv sync --extra dense       # [M3] sentence-transformers, faiss-cpu
uv sync --extra ner         # [M4] spaCy
```

## Từ repo trắng đến số liệu: 4 lệnh

```bash
# 1. Tải HotpotQA distractor, lấy 5000 câu train (seed 42) + toàn bộ dev,
#    gộp context thành một corpus chung, gióng hàng supporting_facts -> doc_id
uv run python scripts/build_corpus.py

# 2. Dựng chỉ mục ngược (biến thể "stem"; thêm "nostem" để so sánh stemming)
uv run python scripts/build_index.py --index stem nostem

# 3. Chạy BM25 trên dev, xuất TREC run file runs/bm25.txt
uv run python scripts/run_retrieval.py --config bm25 --split dev --k 10

# 4. Chấm điểm: P@5, R@10, MAP, nDCG@10, MRR — tổng và theo type / level
uv run python scripts/evaluate.py --run runs/bm25.txt --qrels data/qrels.txt
```

Số liệu nền cho phần đa bước (tỉ lệ câu bridge mà hop 2 không nằm trong top-100):

```bash
uv run python scripts/analyze_hop2.py
```

Kiểm thử: `uv run pytest` (nhóm test `-m data` kiểm tra dữ liệu thật, tự skip nếu chưa chạy bước 1).

## Số liệu nền (dev, 7405 câu, corpus 105.862 đoạn)

Run sâu 100 (`--k 100`); P@5, R@10, nDCG@10 không phụ thuộc độ sâu, MAP/MRR tính trên top-100.

| Cấu hình | P@5 | R@10 | MAP | nDCG@10 | MRR | latency TB |
|---|---|---|---|---|---|---|
| tfidf (lnc.ltc) | 0.2672 | 0.7689 | 0.5893 | 0.6825 | 0.8041 | 20 ms |
| bm25_nostem (k1=1.2, b=0.75) | 0.2729 | 0.7721 | 0.6220 | 0.7095 | 0.8495 | 11 ms |
| **bm25** (k1=1.2, b=0.75) | **0.2777** | **0.7833** | **0.6360** | **0.7229** | **0.8617** | 19 ms |

BM25 theo loại câu: bridge R@10 = 0.7513, comparison R@10 = 0.9109.
Lưu ý: toàn bộ dev của HotpotQA là `level=hard`; tách theo level chỉ có ý nghĩa trên `--split train`.

**Lý do cần đa bước** (`analyze_hop2.py`, BM25, top-100, 5631/5918 câu bridge xác định được hop 2):

- Hop 2 **không** nằm trong top-100 của câu hỏi gốc: **19.73%**
- Đối chứng: hop 1 không nằm trong top-100: 3.59%

## Truy xuất thưa [M2 – Việt]

TF-IDF (SMART), BM25, dò k1/b, skip pointer, champion list, chỉ mục phân tầng — tự cài toàn bộ.
Mọi mô hình viết dưới dạng `score(d,q) = Σ_t w_q(t) · impact_t(d)` (`retrieval/weighting.py`),
nên champion list / phân tầng dùng được cho cả BM25 lẫn mọi sơ đồ SMART.

```bash
uv run python scripts/sparse_experiments.py tune      # dò k1/b trên train (~3 phút)
uv run python scripts/sparse_experiments.py smart     # 7 sơ đồ SMART + BM25 trên dev
uv run python scripts/sparse_experiments.py champion  # r ∈ {50..1000} × {chấm lại, xấp xỉ}
uv run python scripts/sparse_experiments.py tiered    # fractions × {per_term, global} × quy tắc dừng
uv run python scripts/sparse_experiments.py skip      # benchmark skip pointer
# hoặc: ... all     | thêm --limit 100 để chạy thử nhanh
uv run python scripts/run_retrieval.py --config bm25_champion --split dev --k 100   # dùng như mọi cấu hình
```

Bảng kết quả (CSV/JSON) được commit trong `runs/sparse/`; run file TREC và log đầy đủ chỉ có trên
máy chạy (mỗi run ~40 MB, tạo lại bằng các lệnh trên). Mọi số dưới đây: dev 7405 câu, run sâu 100.

**Dò k1/b** (5000 câu train, 96 cấu hình k1 ∈ {0.3..2.4} × b ∈ {0..1}, tối ưu R@100; `bm25_grid.csv`):
tốt nhất **k1 = 0.9, b = 1.0** (train R@100 0.9099 so với 0.9070 của mặc định 1.2/0.75) → cấu hình
`bm25_tuned`. Trên dev: R@100 0.9076 → **0.9099**, bridge R@100 0.8870 → 0.8900; nDCG@10 giảm nhẹ
0.7229 → 0.7214 (đánh đổi do tối ưu recall). Bề mặt khá phẳng quanh đỉnh: b cao luôn có lợi cho
recall, k1 lớn có hại (80% postings có tf = 1 ⇒ bão hoà tf sớm là hợp lý).

**So sánh mô hình trọng số** (`smart.csv`):

| Cấu hình | R@10 | R@100 | MAP | nDCG@10 | MRR |
|---|---|---|---|---|---|
| lnc.ltc (= Lnc.ltc) | 0.7689 | 0.9086 | 0.5893 | 0.6825 | 0.8041 |
| anc.atc | 0.7666 | 0.9085 | 0.5847 | 0.6785 | 0.7992 |
| ltc.ltc | 0.7443 | 0.8968 | 0.5658 | 0.6598 | 0.7872 |
| bnc.btc | 0.7265 | 0.8934 | 0.5400 | 0.6338 | 0.7535 |
| nnc.ntc | 0.6934 | 0.8678 | 0.5298 | 0.6190 | 0.7506 |
| lnc.lnc (không idf) | 0.6492 | 0.8410 | 0.4860 | 0.5729 | 0.6988 |
| bm25 (1.2/0.75) | 0.7833 | 0.9076 | **0.6360** | **0.7229** | **0.8617** |
| **bm25_tuned** (0.9/1.0) | **0.7858** | **0.9099** | 0.6339 | 0.7214 | 0.8568 |

idf là thành phần quan trọng nhất (lnc.lnc tụt 6.8 điểm R@100); tf log/augmented tốt hơn tf thô.
Lnc.ltc trùng hệt lnc.ltc là đúng lý thuyết: hệ số 1/(1+log ave tf) là hằng số theo tài liệu,
bị triệt tiêu khi chuẩn hoá cosine. BM25 hơn tf-idf tốt nhất ~4 điểm nDCG@10 / MRR.

**Champion list** (IIR 7.1.3; `champion.csv`) — xếp postings theo *impact* BM25 thay vì tf thô
(80% postings có tf = 1). "Chấm lại" = tính điểm chính xác cho ứng viên bằng postings đầy đủ
tra qua skip pointer; overlap = tỉ lệ trùng top-k với BM25 vét cạn:

| Cấu hình | R@100 | nDCG@10 | overlap@100 | latency TB | postings chấm/câu |
|---|---|---|---|---|---|
| vét cạn (bm25_tuned) | 0.9099 | 0.7214 | 1 | 18.7 ms | 30 585 |
| r = 50, chấm lại | 0.7851 | 0.6784 | 0.42 | 2.7 ms | 402 |
| r = 200, chấm lại | 0.8840 | 0.7157 | 0.75 | 4.4 ms | 1 480 |
| **r = 500, chấm lại** (`bm25_champion`) | **0.9047** | **0.7207** | 0.92 | **8.1 ms** | 3 390 |
| r = 1000, chấm lại | 0.9097 | 0.7215 | 0.98 | 13.4 ms | 6 108 |
| r = 500, xấp xỉ | 0.8520 | 0.6463 | 0.53 | 2.8 ms | 3 390 |

r = 500: nhanh hơn 2.3×, chấm ít hơn 9× postings, mất 0.5 điểm R@100. Bước chấm lại chính xác
đáng giá +5…+13 điểm nDCG@10 so với chỉ dùng điểm trong champion list. Mất mát khi r nhỏ dồn vào
câu bridge (r = 200: bridge R@100 0.859 so với 0.890).

**Chỉ mục phân tầng** (IIR 7.2.1; `tiered.csv`): **thua champion list ở mọi điểm vận hành.**
Với quy tắc dừng của IIR (xuống tầng khi < k ứng viên), tầng 1 luôn đủ 100 ứng viên nhờ term phổ biến
⇒ không bao giờ xuống tầng 2, R@100 chỉ 0.54–0.85. Nâng ngưỡng lên 5000 ứng viên kéo R@100 về
0.85–0.87 nhưng latency (18–22 ms) không còn nhanh hơn vét cạn. Nguyên nhân: tầng chia theo tỉ lệ
(hoặc ngưỡng toàn cục) cắt cả postings của term hiếm — đúng loại term mang thông tin — trong khi
champion list giữ nguyên mọi term có df ≤ r.

**Skip pointer** (IIR 2.3; `skip.csv`, 2000 cặp term thật + 4532 lần tra ứng viên top-1000; 0 sai khác
so với không skip ở mọi cấu hình):

| Bước | AND 2 term: số so sánh | Tra ứng viên: số so sánh |
|---|---|---|
| không skip | 100% | 100% |
| √P | 84.4% | 100.8% |
| 4 | 70.8% | 80.8% |
| **8** | **67.7%** | **76.8%** |
| 16 | 71.0% | 83.7% |
| 64 | 81.2% | 98.6% |

Bước nhỏ cố định (8) tốt hơn √P trên corpus này (postings đa số ngắn, ứng viên dày). Lưu ý trung thực:
trong Python, giảm phép so sánh **không** chuyển thành giảm thời gian (nhánh skip tốn hơn phép so sánh
tiết kiệm được: 618 ms so với 539 ms) — lợi ích thời gian của skip pointer cần cài đặt mức thấp (C/numpy)
hoặc khi postings nằm trên đĩa. Bước skip khi chấm lại chỉnh được qua `pruning.skip_step` trong config.

**Minh chứng kiểm thử** (mỗi bước có cổng kiểm tra; số liệu kiểm tra nằm trong test tự động và các file
`runs/sparse/byte_check.log`, `factory_check.log`, `tune_best.json`, `skip.csv`):
- Tách module không đổi kết quả: run `bm25`, `tfidf`, `bm25_nostem` mới **trùng từng byte** với run cũ
  (md5 + `cmp` ghi trong `byte_check.log`, kể cả khi chạy qua `factory` mới).
- 10 sơ đồ SMART khớp cài đặt tham chiếu tính theo bảng 6.15 IIR (`tests/test_weighting.py`).
- Skip: 0 sai khác trên 3000 cặp term thật × 4 bước và 17 928 lần tra; property test trên dãy ngẫu nhiên.
- Lưới numpy: top-100 tại (1.2, 0.75) trùng run BM25 gốc ở **7405/7405** câu dev; 0/300 sai khác tại điểm tốt nhất.
- Champion r ≥ df max và tiered một tầng ⇒ **trùng từng bit** với vét cạn (`tests/test_pruning.py`);
  champion r = 1000 trên dev cho số liệu bằng vét cạn.

## Dữ liệu sinh ra (`data/`, không commit)

| File | Nội dung |
|---|---|
| `corpus.jsonl` | `doc_id`, `title`, `sentences`, `text` — khử trùng lặp theo title, doc_id liên tục 0..N-1 |
| `questions.jsonl` | `qid`, `split` (train/dev), `question`, `answer`, `type`, `level`, `gold_doc_ids`, `supporting_facts` [{title, sent_id, doc_id}] |
| `qrels.txt` | TREC qrels `qid 0 doc_id 1`, mỗi câu 2 đoạn vàng |
| `alignment_report.json` | Thống kê gióng hàng + danh sách nhiễu nhãn gốc bị loại |
| `index/<tên>.pkl` | Chỉ mục ngược đã pickle |

**Gióng hàng nhãn vàng.** `build_corpus.py` dừng ngay (`AlignmentError`) nếu bất kỳ
supporting fact nào không tìm được doc_id, hoặc câu tại (doc_id, sent_id) trong corpus
khác câu gốc. Hai trường hợp sent_id vượt quá số câu *ngay trong context gốc* của
HotpotQA là nhiễu nhãn nguồn: fact bị loại, ghi vào `alignment_report.json`; đoạn văn
vẫn là đoạn vàng. Các title trùng nhưng khác nội dung (127 lần) chỉ khác khoảng trắng
(NBSP), số câu bằng nhau.

## Cấu trúc và phân công

```
config/default.yaml   tham số dùng chung (dữ liệu, chỉ mục, baseline, đánh giá)
config/parts/         cấu hình riêng từng phần, tự gộp: sparse.yaml [M2], dense.yaml [M3], ...
src/hmr/
  common/             interfaces.py (HỢP ĐỒNG CHUNG), tokenizer.py, io.py, config.py, log.py
  data/               hotpot.py, corpus.py, alignment.py, hops.py
  index/       [M1]   inverted.py, compression.py (delta+VByte: TODO)
               [M2]   skip.py (skip pointer)
  retrieval/   [M2]   weighting.py, scoring.py, merge.py, bm25.py, tfidf.py, bm25_grid.py,
                      pruning.py, champion.py, tiered.py, factory.py
               [M3]   dense, hybrid: TODO
  multihop/    [M4]   trích thực thể, sinh truy vấn hop 2: TODO
  eval/        [M5]   metrics.py
scripts/              build_corpus, build_index, run_retrieval, evaluate, analyze_hop2,
                      sparse_experiments [M2]
tests/
```

**Quy ước code, đặc tả dữ liệu, bản đồ sở hữu file, cách cắm phần mới và quy trình git:** xem
[HUONG_DAN_THANH_VIEN.md](HUONG_DAN_THANH_VIEN.md) — nguồn duy nhất cho các nội dung này.

## Ghi chú cài đặt

- **BM25**: `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, term lặp trong truy vấn nhân theo qtf.
  Điểm cộng dồn bằng phép trộn hai con trỏ trên postings đã sắp (`retrieval/merge.py`),
  term df nhỏ xử lý trước.
- **tf-idf lnc.ltc**: chuẩn cosine tài liệu tính một lần khi khởi tạo (duyệt `vocabulary()`).
- **Chỉ mục**: mặc định chỉ mục `title + text` (`include_title`), vị trí từ tính trên dãy
  token sau tokenize.
- **Chấm điểm**: câu hỏi của split không có trong run được tính 0 điểm (pytrec_eval mặc
  định bỏ qua chúng).
