# hotpot-multihop

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
config/default.yaml   mọi tham số (k1, b, top_k, đường dẫn, biến thể chỉ mục)
src/hmr/
  common/             interfaces.py (HỢP ĐỒNG CHUNG), tokenizer.py, io.py, config.py, log.py
  data/               hotpot.py, corpus.py, alignment.py, hops.py
  index/       [M1]   inverted.py, compression.py (delta+VByte: TODO), skip.py (skip pointer: TODO)
  retrieval/   [M2]   merge.py, tfidf.py, bm25.py, factory.py     [M3] dense, hybrid: TODO
  multihop/    [M4]   trích thực thể, sinh truy vấn hop 2: TODO
  eval/        [M5]   metrics.py
scripts/              build_corpus, build_index, run_retrieval, evaluate, analyze_hop2
tests/
```

### Quy ước bắt buộc

- **Interface** (`src/hmr/common/interfaces.py`): `Index`, `Retriever`, `MultiHopRetriever`,
  `RetrievalResult` / `RankedPassage` / `HopTrace`. Đổi chữ ký phải thống nhất cả nhóm.
- **Tokenizer** duy nhất: `hmr.common.tokenizer.tokenize(text, stem=True, remove_stopwords=True)`
  (lowercase → `\w+` → stopword NLTK → Porter). Truy vấn luôn tokenize qua
  `index.analyze(...)` để khớp đúng tham số lúc dựng chỉ mục.
- **Output** mọi cấu hình: TREC run `qid Q0 doc_id rank score run_name`.
  Cấu hình đa bước bọc vào `MultiHopRetriever` và trả `RetrievalResult`;
  `run_retrieval.py` ghi run từ `RetrievalResult.as_ranking()`.
- **Không hardcode tham số**: thêm khoá vào `config/default.yaml`, thêm cấu hình truy xuất
  mới vào mục `retrievers` rồi chạy `run_retrieval.py --config <tên>`.
- `logging` thay cho `print`, `tqdm` cho vòng lặp dài.

### Hướng dẫn cho từng thành viên

- **M1**: cài 4 hàm trong `index/compression.py` và 2 hàm trong `index/skip.py`; các test
  `xfail` trong `tests/test_compression_skip.py` sẽ chuyển sang XPASS. Bật nén bằng
  `codec: delta_vbyte` trong `indexes` — không cần sửa `inverted.py`.
- **M3**: thêm `retrieval/dense.py`, `retrieval/hybrid.py` cài `Retriever`, rồi thêm nhánh
  trong `retrieval/factory.py::build_retriever`.
- **M4**: cài `MultiHopRetriever` trong `multihop/`, ghi `HopTrace.bridge_entities`.
  Số liệu nền ở `scripts/analyze_hop2.py` là đường cơ sở cần vượt.
- **M5**: mở rộng `eval/metrics.py` (vd. đánh giá theo cặp đoạn vàng, recall mức câu từ
  `supporting_facts`).

## Ghi chú cài đặt

- **BM25**: `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`, term lặp trong truy vấn nhân theo qtf.
  Điểm cộng dồn bằng phép trộn hai con trỏ trên postings đã sắp (`retrieval/merge.py`),
  term df nhỏ xử lý trước.
- **tf-idf lnc.ltc**: chuẩn cosine tài liệu tính một lần khi khởi tạo (duyệt `vocabulary()`).
- **Chỉ mục**: mặc định chỉ mục `title + text` (`include_title`), vị trí từ tính trên dãy
  token sau tokenize.
- **Chấm điểm**: câu hỏi của split không có trong run được tính 0 điểm (pytrec_eval mặc
  định bỏ qua chúng).
