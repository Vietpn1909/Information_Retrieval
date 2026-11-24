# Truy xuất dày & lai [M3 – Khiêm]

> Tổng quan dự án: [README](../README.md). Quy ước chung: [HUONG_DAN_THANH_VIEN.md](HUONG_DAN_THANH_VIEN.md).

Bi-encoder **BGE-small** + **FAISS**, truy xuất **lai** BM25 + dense (RRF / tổ hợp tuyến tính) và
**xếp hạng lại** bằng cross-encoder. Mọi cấu hình đi qua `factory.build_retriever` như phần thưa,
xuất TREC run và chấm bằng harness chung. Mọi tham số dò trên **train** (5000 câu), chỉ báo cáo
trên **dev** (7405 câu).

## Chạy

```bash
uv sync --extra dense
uv run python scripts/build_dense.py --model bge_small minilm      # mã hoá kho -> data/dense/*.npy
uv run python scripts/run_retrieval.py --config rerank_hybrid --split dev --k 100
uv run python scripts/evaluate.py --run runs/rerank_hybrid.txt

# Thí nghiệm (bảng trong runs/dense/)
uv run python scripts/dense_experiments.py tune         # dò gộp hybrid trên train   (~5 phút GPU)
uv run python scripts/dense_experiments.py rerank_tune  # dò reranker trên train     (~7 phút GPU)
uv run python scripts/dense_experiments.py ann          # flat vs HNSW vs IVF        (~3 phút)
uv run python scripts/dense_experiments.py eval         # bảng chính trên dev        (~40 phút GPU)
# hoặc: ... all     | thêm --limit 200 để chạy thử nhanh
```

**Thời gian.** Trên GPU (RTX 4090) mã hoá 105 862 đoạn mất khoảng 1 phút mỗi model. Trên CPU
mất 20–40 phút. Cross-encoder `bge-reranker-base` chấm 100 ứng viên mất khoảng 130 ms/câu trên
GPU, và chậm hơn nhiều lần trên CPU. Muốn chạy GPU thì đặt `dense.device: cuda` trong
`config/parts/dense.yaml` (mặc định `cpu`). Các bảng dưới đây chạy trên GPU, còn BM25 và FAISS
chạy trên CPU.

**Dữ liệu sinh ra** (không commit): `data/dense/<model>.npy` là ma trận float32 kích thước
(105 862 × 384), đã chuẩn hoá L2, trong đó dòng i chính là `doc_id = i`. File `.json` đi kèm ghi
tên model và **md5 của corpus**. Khi nạp, chương trình báo lỗi nếu vectơ được dựng từ corpus khác
hoặc model khác.

## Cài đặt

| File | Nội dung |
|---|---|
| `retrieval/dense.py` | `DenseRetriever` mã hoá truy vấn (prefix BGE `"Represent this sentence for searching relevant passages: "`) rồi tìm FAISS inner product. Vectơ đã chuẩn hoá nên inner product chính là cosine. Đoạn văn được mã hoá dưới dạng `"{title}. {text}"`, giống chỉ mục thưa (title + text). FAISS có ba loại: `flat` / `hnsw` / `ivf`. |
| `retrieval/hybrid.py` | `rrf_fuse`: Σ wᵢ/(rrf_k + rankᵢ). `linear_fuse`: Σ wᵢ·norm(sᵢ), với norm là `minmax` hoặc `zscore` tính trên danh sách ứng viên. `HybridRetriever` lấy `depth` = 200 ứng viên từ mỗi thành phần. Thành phần là tên cấu hình bất kỳ, nên có thể gộp cả `bm25_champion`. |
| `retrieval/rerank.py` | `RerankRetriever` chấm lại `rerank_depth` ứng viên đầu bằng cross-encoder, phần đuôi giữ nguyên thứ tự base với điểm giảm ngặt. Tuỳ chọn `combine: {rrf_k, ce_weight}` gộp RRF thứ hạng cross-encoder với thứ hạng base. |
| `scripts/build_dense.py` | Mã hoá kho, kiểm tra `doc_id` liên tục. |
| `scripts/dense_experiments.py` | `ann`, `tune`, `rerank_tune`, `eval`. Kết quả mỗi model được cache theo (cấu hình, split), nên hybrid và rerank dùng lại kết quả của thành phần, không tính lại. |
| `config/parts/dense.yaml` | Mọi cấu hình và tham số. |
| `tests/test_dense_retrieval.py` | 13 test chạy bằng bộ mã hoá giả (không tải model): RRF và linear tính tay, rerank, factory, md5 corpus, vectơ thật (`-m data`). |

Cấu hình đăng ký trong `retrievers`: `dense_bge`, `dense_bge_hnsw`, `dense_bge_ivf`, `dense_minilm`,
`hybrid_rrf`, `hybrid_linear`, `rerank_hybrid`, `rerank_bm25`, `rerank_hybrid_minilm`.

## Kết quả chính (dev, 7405 câu, run sâu 100) — `runs/dense/dev.csv`

| Cấu hình | R@10 | R@100 | MAP | nDCG@10 | MRR | Cả 2 vàng @10 | Hop 2 trượt top-100 | Latency |
|---|---|---|---|---|---|---|---|---|
| bm25_tuned (mốc M2) | 0.7858 | 0.9099 | 0.6339 | 0.7214 | 0.8568 | 0.594 | 19.22% | 27 ms |
| dense_minilm | 0.6756 | 0.8132 | 0.5588 | 0.6433 | 0.8099 | 0.424 | 36.67% | 18 ms |
| dense_bge | 0.8069 | 0.9122 | 0.7096 | 0.7763 | 0.9019 | 0.642 | 18.24% | 21 ms |
| hybrid_linear (zscore, α = 0.4) | 0.8597 | 0.9406 | 0.7587 | 0.8238 | 0.9359 | 0.727 | 13.12% | 46 ms |
| **hybrid_rrf** (rrf_k = 20, w = 1 : 1.5) | 0.8527 | **0.9425** | 0.7340 | 0.8049 | 0.9161 | 0.716 | **12.77%** | 46 ms |
| rerank_bm25 (bge-reranker, top-100) | 0.8879 | 0.9099 | 0.8274 | 0.8745 | **0.9650** | 0.781 | 19.22% | 152 ms* |
| rerank_hybrid_minilm (MiniLM, top-50, gộp RRF) | 0.8756 | 0.9425 | 0.7625 | 0.8306 | 0.9360 | 0.755 | 12.77% | 76 ms* |
| **rerank_hybrid** (bge-reranker, top-100) | **0.9028** | **0.9425** | **0.8371** | **0.8823** | 0.9644 | **0.811** | **12.77%** | 182 ms* |

\* Latency đo trên GPU dùng chung với job khác. Latency đo theo từng câu qua `Retriever.search`
trên 200 câu dev, gồm cả bước mã hoá truy vấn. "Cả 2 vàng @10" là tỉ lệ câu có **cả hai** đoạn
vàng trong top-10, tức điều kiện cần để trả lời câu multi-hop. Hop 2 được xác định bằng
`hmr.data.hops.identify_hop2`, áp dụng cho 5631 câu bridge, giống `analyze_hop2.py`.

So với mốc `bm25_tuned`, cấu hình tốt nhất `rerank_hybrid` đạt:
- **R@100 tăng 0.910 → 0.943** (+3.3 điểm, nhờ hybrid; rerank không đổi tập top-100).
- **nDCG@10 tăng 0.721 → 0.882** (+16.1 điểm), MAP tăng +20.3 điểm.
- **Tỉ lệ câu có cả 2 đoạn vàng trong top-10 tăng 59.4% → 81.1%.**
- **Tỉ lệ câu có đoạn hop 2 trượt top-100 giảm 19.22% → 12.77%** (giảm khoảng 1/3). Phần còn lại
  là việc của truy xuất đa bước (M4). `hybrid_rrf` là bộ truy xuất mỗi hop được khuyến nghị cho M4:
  rẻ (46 ms) và có recall cao nhất.

Theo loại câu (R@100 / nDCG@10):

| Cấu hình | bridge | comparison |
|---|---|---|
| bm25_tuned | 0.890 / 0.712 | 0.989 / 0.758 |
| dense_bge | 0.890 / 0.727 | 0.999 / 0.972 |
| hybrid_rrf | 0.928 / 0.771 | 0.999 / 0.941 |
| rerank_hybrid | 0.928 / **0.854** | 0.999 / **0.994** |

Dense đặc biệt mạnh ở câu comparison (nDCG@10 0.972 so với 0.758 của BM25): cả hai thực thể đều
xuất hiện trong câu hỏi, và bi-encoder không bị loãng bởi các từ so sánh như "which", "older",
"both". Câu bridge là chỗ khó còn lại.

### BM25 và dense bổ sung cho nhau (`complementarity.json`)

Top-100 của hai bộ chỉ trùng **19%**, top-10 trùng 33%. Xét 14 810 đoạn vàng của dev:

| | Cả hai tìm thấy | Chỉ BM25 | Chỉ dense | Không bộ nào |
|---|---|---|---|---|
| top-10 | 71.7% | 6.9% | 9.0% | 12.4% |
| top-100 | 86.9% | 4.1% | 4.4% | 4.7% |

Mỗi bộ tìm được khoảng 4% đoạn vàng mà bộ kia bỏ sót. Đây là lý do gộp lai tăng R@100 lên 3 điểm,
dù riêng từng bộ có R@100 gần như bằng nhau (0.910 so với 0.912).

## Dò tham số (train, không dùng dev)

**Gộp hybrid** (`hybrid_tune.csv`, `hybrid_best.json`, 5000 câu, tối ưu R@100, phá hoà bằng nDCG@10):

| Cách gộp | R@100 | nDCG@10 |
|---|---|---|
| chỉ bm25_tuned | 0.9099 | 0.7344 |
| chỉ dense_bge | 0.9176 | 0.7916 |
| RRF rrf_k = 60 (mặc định trong tài liệu gốc), 1 : 1 | 0.9432 | 0.8105 |
| **RRF rrf_k = 20, 1 : 1.5** → `hybrid_rrf` | **0.9438** | 0.8171 |
| linear minmax α = 0.4 | 0.9421 | 0.8301 |
| **linear zscore α = 0.4** → `hybrid_linear` | 0.9427 | **0.8346** |

- R@100 gần như **phẳng** theo rrf_k (0.942–0.944 với rrf_k từ 1 đến 100), nên RRF ổn định và
  không cần dò kỹ. rrf_k nhỏ thì nDCG@10 cao hơn, vì trọng số dồn nhiều hơn vào top đầu.
- Linear có nDCG@10 tốt hơn RRF khoảng 1.8 điểm (dùng được độ lớn điểm, không chỉ thứ hạng), còn
  R@100 thấp hơn một chút. Đỉnh ở α = 0.4 (BM25 0.4, dense 0.6), zscore nhỉnh hơn minmax.

**Xếp hạng lại** (`rerank_tune.csv`, `rerank_best.json`, 1000 câu train, tối ưu nDCG@10). Thử hai
cross-encoder × depth ∈ {10, 20, 50, 100} × {chỉ điểm CE, gộp RRF với base}:

| Base | Reranker | depth | Cách gộp | nDCG@10 | ms/câu (GPU) |
|---|---|---|---|---|---|
| hybrid_rrf | không rerank | – | – | 0.8241 | – |
| hybrid_rrf | ms-marco-MiniLM-L-6 | 50 | chỉ CE | 0.8125 ↓ | 19 |
| hybrid_rrf | ms-marco-MiniLM-L-6 | 50 | RRF k = 20 | **0.8331** | 19 |
| hybrid_rrf | bge-reranker-base | 20 | chỉ CE | 0.8962 | 26 |
| hybrid_rrf | bge-reranker-base | 50 | chỉ CE | 0.9112 | 65 |
| hybrid_rrf | **bge-reranker-base** | **100** | **chỉ CE** | **0.9179** | 130 |
| bm25_tuned | ms-marco-MiniLM-L-6 | 100 | chỉ CE | 0.8037 | 37 |
| bm25_tuned | bge-reranker-base | 100 | chỉ CE | 0.8968 | 127 |

Nhận xét:
- **Cross-encoder MS MARCO nhỏ làm giảm chất lượng hybrid** (0.824 → 0.813), dù nó giúp BM25
  (0.739 → 0.804). Model này chấm từng đoạn độc lập với câu hỏi, nên đoạn hop 2 (ít trùng nội dung
  với câu hỏi) bị đẩy xuống, trong khi hybrid vốn đã xếp đoạn đó cao. Gộp RRF thứ hạng CE với
  thứ hạng base khắc phục được điểm này (0.833). Vì vậy cấu hình `rerank_hybrid_minilm` dùng
  `combine`.
- **bge-reranker-base mạnh hơn hẳn** (+9.4 điểm nDCG@10 trên hybrid). Với model này, gộp với base
  lại kém hơn chỉ dùng điểm CE. depth càng lớn càng tốt nhưng lợi ích giảm dần (50 → 100 chỉ thêm
  0.7 điểm với chi phí gấp đôi). Nếu cần nhanh hơn thì dùng `rerank_depth: 50`.
- Rerank không làm thay đổi R@100, vì chỉ sắp lại các ứng viên đã có. Recall phụ thuộc vào bộ
  sinh ứng viên, nên base phải là hybrid.

## Tìm kiếm xấp xỉ (ANN) — `ann.csv`

dense_bge trên dev, đo từng truy vấn một với vectơ đã mã hoá sẵn (bước mã hoá truy vấn mất thêm
khoảng 6 ms trên GPU). FAISS chạy CPU. Overlap là tỉ lệ trùng top-100 với flat (chính xác).

| Chỉ mục | Tham số | R@100 | nDCG@10 | overlap@100 | Tìm (ms/câu) | Dựng |
|---|---|---|---|---|---|---|
| flat (chính xác) | – | 0.9122 | 0.7763 | 1 | 13.7 | 0.1 s |
| HNSW (M = 32) | efSearch 64 | 0.8804 | 0.7496 | 0.878 | 0.54 | 11.6 s |
| HNSW | efSearch 128 | 0.8976 | 0.7647 | 0.948 | 0.91 | |
| HNSW | **efSearch 256** | **0.9070** | **0.7725** | **0.981** | **1.50** | |
| IVF (nlist 1024) | nprobe 32 | 0.8199 | 0.7063 | 0.835 | 0.65 | 1.8 s |
| IVF | nprobe 128 | 0.8889 | 0.7600 | 0.959 | 2.09 | |

- HNSW efSearch 256 nhanh hơn flat **9 lần**, chỉ mất 0.5 điểm R@100. Trên cùng mức chính xác,
  HNSW luôn nhanh hơn IVF.
- Kho chỉ có 106 nghìn đoạn, nên tìm flat (14 ms) không phải nút cổ chai so với mã hoá truy vấn
  và BM25. Vì vậy các cấu hình chính dùng `flat` (kết quả chính xác, tất định). ANN chỉ cần khi
  kho lớn hơn nhiều.

## Minh chứng kiểm thử

- `uv run pytest` xanh, gồm 13 test M3 và test dữ liệu thật: vectơ có đúng 105 862 dòng, chuẩn
  hoá L2, md5 corpus khớp `data.checksums`.
- Harness của `dense_experiments.py` cho `bm25_tuned` trên dev đúng bằng mốc M2
  (R@100 0.9099, nDCG@10 0.7214, hop 2 trượt 19.22%).
- Đường chạy chính thức `run_retrieval.py --config hybrid_rrf` (CPU, máy khác) so với run do
  `dense_experiments.py` sinh trên GPU, trên 300 câu dev: **300/300 trùng top-10, 298/300 trùng
  toàn bộ top-100**. Hai câu lệch chỉ đổi thứ tự đoạn hoà điểm do sai số float giữa GPU và CPU.
