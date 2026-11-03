# Hướng dẫn cho thành viên

Đọc file này **trước khi bắt đầu code**. Mục tiêu: 5 người làm song song mà không ghi đè, không
lệch dữ liệu, không conflict khi merge. Tổng quan dự án xem [README](../README.md); chi tiết cài đặt
và số liệu phần đã xong xem [CHI_TIET.md](CHI_TIET.md).

## 0. Bảy quy tắc vàng

1. **Chỉ sửa file trong vùng của mình** (bảng sở hữu ở [mục 4](#4-bản-đồ-sở-hữu-file)). Cần đổi file
   của người khác → nhắn người đó, không tự sửa.
2. **Không sửa** pipeline dữ liệu, tokenizer, `interfaces.py`, định dạng I/O nếu chưa thống nhất cả nhóm
   — sửa là doc_id / kết quả của mọi người lệch nhau.
3. **Cấu hình riêng để trong `config/parts/<phần>.yaml`**, không sửa `config/default.yaml`
   ([mục 5](#5-cắm-phần-mới-vào-không-sửa-file-chung)).
4. **Thêm loại truy xuất mới bằng cách tạo file mới** — `factory.py` và `run_retrieval.py` đã đăng ký
   sẵn chỗ cho dense / hybrid / rerank / multihop, không cần sửa.
5. **Dev chỉ để báo cáo.** Mọi tinh chỉnh tham số / huấn luyện dùng split `train`.
6. **Không commit `data/` và run file TREC** (đã có trong `.gitignore`); chỉ commit bảng kết quả nhỏ.
7. **Mỗi người một nhánh, merge qua Pull Request**, `uv run pytest` phải xanh trước khi merge.

## 1. Thiết lập lần đầu (~5 phút)

```bash
git clone https://github.com/Vietpn1909/Information_Retrieval.git
cd Information_Retrieval/hotpot-multihop
uv sync                                              # + --extra dense (Khiêm) / --extra ner (Cường)
uv run python scripts/build_corpus.py                # tải HotpotQA, dựng data/ (~2 phút)
uv run python scripts/build_index.py --index stem nostem   # chỉ mục ngược (~30 giây)
uv run pytest                                        # PHẢI xanh, kể cả nhóm test dữ liệu thật
```

`uv run pytest` sẽ tự kiểm tra **md5 dữ liệu bạn vừa dựng trùng bản chuẩn** (mục 2.5). Nếu fail
ở `test_data_matches_reference_checksum` → dữ liệu của bạn lệch với cả nhóm, đừng chạy thí nghiệm,
báo lại ngay.

## 2. Dữ liệu

### 2.1 Nguồn và cách dựng

| Mục | Giá trị (trong `config/default.yaml`, khoá `data`) |
|---|---|
| Nguồn | HuggingFace `hotpotqa/hotpot_qa`, cấu hình `distractor` |
| Phiên bản | `hf_revision: 1908d6afbbead072334abe2965f91bd2709910ab` (đã pin — mọi máy tải đúng bản này) |
| Câu hỏi | 5000 câu lấy mẫu từ train (`seed: 42`) + **toàn bộ** 7405 câu dev (split `validation`) |
| Kho đoạn văn | Gộp 10 đoạn ngữ cảnh của mọi câu trên thành **một kho chung**, khử trùng lặp theo `title` |

Toàn bộ `data/` **không cần bàn giao**: dựng lại từ HuggingFace bằng 2 lệnh ở mục 1 cho kết quả
trùng từng byte (đã kiểm chứng bằng md5).

### 2.2 Thống kê

| | Train (mẫu) | Dev |
|---|---|---|
| Số câu hỏi | 5 000 | 7 405 |
| bridge / comparison | 4 032 / 968 | 5 918 / 1 487 |
| easy / medium / hard | 976 / 3 168 / 856 | 0 / 0 / 7 405 (dev **chỉ có hard**) |
| Đoạn vàng mỗi câu | đúng 2 | đúng 2 |

Kho: **105 862 đoạn**; chỉ mục `stem`: 192 823 term, độ dài đoạn trung bình 57,6 token (sau tokenize).

### 2.3 Các file trong `data/`

**`corpus.jsonl`** — một đoạn văn mỗi dòng:

```json
{"doc_id": 46186, "title": "Kiss and Tell (1945 film)",
 "sentences": ["Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.", " In the film, two teenage girls ...", "..."],
 "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, tw..."}
```

- `sentences`: giữ nguyên như HotpotQA (có dấu cách đầu câu) — `sent_id` là chỉ số trong mảng này.
- `text`: các câu nối lại, đã chuẩn hoá khoảng trắng. **Không chứa title.**

**`questions.jsonl`** — một câu hỏi mỗi dòng:

```json
{"qid": "5a8c7595554299585d9e36b6", "split": "dev",
 "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
 "answer": "Chief of Protocol", "type": "bridge", "level": "hard",
 "gold_doc_ids": [46186, 46181],
 "supporting_facts": [{"title": "Kiss and Tell (1945 film)", "sent_id": 0, "doc_id": 46186},
                      {"title": "Shirley Temple", "sent_id": 0, "doc_id": 46181},
                      {"title": "Shirley Temple", "sent_id": 1, "doc_id": 46181}]}
```

- `split`: `"train"` hoặc `"dev"`. `type`: `"bridge"` / `"comparison"`. `level`: `"easy"` / `"medium"` / `"hard"`.
- `supporting_facts` **đã gióng hàng sẵn sang `doc_id`** — dùng trực tiếp cho đánh giá mức câu.

**`qrels.txt`** — TREC qrels nhị phân, 2 dòng mỗi câu: `qid 0 doc_id 1`.

**`alignment_report.json`** — thống kê gióng hàng: 29 916/29 916 fact (100%), 2 fact nhiễu nhãn gốc bị loại.

**`index/stem.pkl`, `index/nostem.pkl`** — `InvertedIndex` (pickle), nạp bằng
`hmr.retrieval.factory.load_index(cfg, "stem")`. Chỉ mục nội dung **`title + text`**.

### 2.4 Bất biến — được phép dựa vào

- `doc_id` là số nguyên liên tục `0..105861` **= số thứ tự dòng** trong `corpus.jsonl`.
- `title` là duy nhất; `corpus[doc_id]["title"]` khớp `title` trong `supporting_facts`.
- Mỗi câu hỏi có **đúng 2** `gold_doc_ids`; `qrels.txt` khớp `gold_doc_ids`.
- `qid` = `id` gốc của HotpotQA (chuỗi hex 24 ký tự).
- Mọi kết quả truy xuất dùng **đúng `doc_id` này** — không tự đánh lại số (vd. thứ tự vectơ trong FAISS
  phải ánh xạ về `doc_id`).

### 2.5 Mã kiểm tra (md5)

| File | md5 |
|---|---|
| `corpus.jsonl` | `6a139d161d2206c9648e05e98bf7f024` |
| `questions.jsonl` | `8babcefff51d5115c77e6debd1f7ecbb` |
| `qrels.txt` | `6bc5a4c6fc74d1d481ce31d1643c7824` |

Lưu trong `config/default.yaml` (`data.checksums`), kiểm tra tự động bằng `uv run pytest -m data`.

### 2.6 Đặc điểm cần biết

- 2 câu có fact nhiễu nhãn gốc (sent_id vượt số câu): `5ab6460b5542995eadeeff96`, `5ae61bfd5542992663a4f261`
  — fact đó bị loại, đoạn vẫn là đoạn vàng.
- 127 lần gặp cùng title với nội dung chỉ khác khoảng trắng (NBSP) — giữ bản đầu.
- **Dev toàn câu `hard`** — tách theo `level` chỉ có nghĩa trên train.
- 80% postings có tf = 1 (đoạn ngắn) — các phương pháp dựa vào tf thô ít tác dụng.

### 2.7 Quy tắc dùng dữ liệu

- **Không ghi đè** bất kỳ file nào trong `data/`. Sản phẩm phụ của phần mình (vectơ dense, cache NER...)
  để trong thư mục con riêng: `data/dense/`, `data/multihop/`, ... (đã bị gitignore).
- Tinh chỉnh / huấn luyện: chỉ dùng `split == "train"`. Báo cáo: `split == "dev"`.
- Cần thay đổi cách dựng dữ liệu → **thống nhất cả nhóm**, cập nhật `data.checksums`, mọi người dựng lại.

## 3. Những gì đã có — dùng lại, đừng viết lại

**Đóng băng** (đổi phải thống nhất cả nhóm): `common/interfaces.py`, `common/tokenizer.py`,
`common/io.py`, toàn bộ `data/` + `scripts/build_corpus.py`.

| Cần gì | Dùng | Ghi chú |
|---|---|---|
| Hợp đồng chung | `hmr.common.interfaces`: `Index`, `Retriever`, `MultiHopRetriever`, `RetrievalResult`, `RankedPassage`, `HopTrace` | `Retriever.search(query, k) -> [(doc_id, score)]` giảm dần |
| Tách từ | `hmr.common.tokenizer.tokenize(text, stem=True, remove_stopwords=True)` | Truy vấn trên chỉ mục: dùng `index.analyze(text)` |
| Đọc/ghi | `hmr.common.io`: `read_jsonl`, `write_trec_run`, `read_trec_run`, `read_qrels` | |
| Cấu hình | `hmr.common.config.load_config()`, `retriever_config(cfg, name)` | tự gộp `config/parts/*.yaml` |
| Chỉ mục ngược | `InvertedIndex`: `postings`, `df`, `doc_len`, `positional_postings`, `positions`, `analyze` | `hmr.index.inverted` |
| Truy xuất thưa có sẵn | `build_retriever(cfg, "bm25_tuned")` — hoặc `bm25`, `tfidf`, `bm25_champion` | `hmr.retrieval.factory` |
| Chạy + ghi run | `scripts/run_retrieval.py --config <tên>` | mọi loại (kể cả đa bước) đi qua `build_system` |
| Chấm điểm | `scripts/evaluate.py --run ...`; `hmr.eval.metrics.evaluate_per_query`, `aggregate` | câu thiếu trong run tính 0 điểm |
| Đoạn hop 2 của câu bridge | `hmr.data.hops.identify_hop2` | dùng cho phân tích đa bước |
| Giao postings / skip | `hmr.index.skip`: `intersect`, `intersect_with_skips`, `SkipPostings` | |

**Mốc cần vượt** (dev, run sâu 100): `bm25_tuned` R@100 = **0.9099**, nDCG@10 = **0.7214**;
**19,73%** câu bridge có đoạn hop 2 nằm ngoài top-100 BM25 (`scripts/analyze_hop2.py`).

## 4. Bản đồ sở hữu file

| Vùng | File | Chủ | Người khác |
|---|---|---|---|
| Chung | `common/*`, `data/*` (code), `scripts/build_corpus.py`, `pyproject.toml`, `uv.lock`, `config/default.yaml` | Cả nhóm | Chỉ sửa khi đã thống nhất |
| Chỉ mục | `index/inverted.py`, `index/compression.py`, `scripts/build_index.py`, `tests/test_inverted.py`, `tests/test_compression_skip.py`, `config/parts/index.yaml` | **Nguyên** | Chỉ đọc |
| Truy xuất thưa | `index/skip.py`, `retrieval/{weighting,scoring,merge,bm25,tfidf,bm25_grid,pruning,champion,tiered,factory}.py`, `scripts/sparse_experiments.py`, `config/parts/sparse.yaml`, `runs/sparse/` | **Việt** | Chỉ đọc |
| Dày & lai | `retrieval/dense.py`, `retrieval/hybrid.py`, `retrieval/rerank.py` *(tạo mới)*, `config/parts/dense.yaml`, `data/dense/`, `runs/dense/` | **Khiêm** | Chỉ đọc |
| Đa bước | `multihop/*` *(tạo mới)*, `config/parts/multihop.yaml`, `scripts/analyze_hop2.py`, `runs/multihop/` | **Cường** | Chỉ đọc |
| Đánh giá & demo | `eval/*`, `scripts/evaluate.py`, các script vẽ đồ thị / demo *(tạo mới)*, `config/parts/eval.yaml` | **Vinh** | Chỉ đọc |
| Tài liệu | `README.md`: mỗi người chỉ sửa **dòng tiến độ của mình**; tài liệu riêng: `docs/<phần>.md` | Mỗi người | |

Test: mỗi người viết file test riêng `tests/test_<phần>_*.py` — không thêm vào file test của người khác.

## 5. Cắm phần mới vào (không sửa file chung)

### 5.1 Cấu hình riêng: `config/parts/<phần>.yaml`

Mọi file `config/parts/*.yaml` được gộp tự động vào `config/default.yaml`. Trùng tên khoá với file khác
→ báo lỗi ngay (không âm thầm ghi đè). Ví dụ `config/parts/dense.yaml`:

```yaml
paths:
  dense_dir: data/dense          # khoá MỚI trong paths — được phép
retrievers:
  dense_bge:                     # tên MỚI — trùng tên có sẵn sẽ báo lỗi
    type: dense
    model: BAAI/bge-small-en-v1.5
dense:                           # mục riêng của phần mình
  batch_size: 64
```

### 5.2 Truy xuất mới — Khiêm (`dense`, `hybrid`, `rerank`)

`factory.py` đã đăng ký sẵn `type: dense → hmr.retrieval.dense:build` (tương tự `hybrid`, `rerank`).
Chỉ cần tạo file:

```python
# src/hmr/retrieval/dense.py
from collections.abc import Mapping
from typing import Any
from hmr.common.config import Config

class DenseRetriever:                       # cài đặt hmr.common.interfaces.Retriever
    def search(self, query: str, k: int) -> list[tuple[int, float]]: ...

def build(cfg: Config, rcfg: Mapping[str, Any]) -> DenseRetriever:
    """Đọc tham số từ rcfg (khối retrievers.<tên>) và cfg['dense']."""
    ...
```

Rồi `uv run python scripts/run_retrieval.py --config dense_bge --split dev --k 100` → run file +
chấm bằng `evaluate.py` như mọi cấu hình. Hybrid cần BM25: gọi `build_retriever(cfg, "bm25_tuned")`.

### 5.3 Đa bước — Cường (`type: multihop...`)

Mọi `type` bắt đầu bằng `multihop` (vd. `multihop_entity`, `multihop_iter`) được chuyển tới
`hmr.multihop.factory:build`, và **không** bị bọc adapter — builder phải trả `MultiHopRetriever`:

```python
# src/hmr/multihop/factory.py
def build(cfg, rcfg):          # rcfg["type"] cho biết chiến lược nào → tự rẽ nhánh bên trong
    base = build_retriever(cfg, rcfg.get("base", "bm25_tuned"))   # bộ truy xuất cho từng hop
    ...
    return MyMultiHop(base, ...)   # search(question, k) -> RetrievalResult
```

- Điền `RetrievalResult.hops` (mỗi hop một `HopTrace` với `query`, `retrieved_ids`, `bridge_entities`)
  — Vinh dùng để phân tích lỗi.
- `qid` để `""` — `run_retrieval.py` tự gán.
- Model spaCy cần cài thêm vào môi trường — thống nhất cách cài (ghi vào tài liệu của phần mình).

### 5.4 Nén chỉ mục — Nguyên

- Cài 4 hàm trong `index/compression.py`; các test `xfail` trong `tests/test_compression_skip.py` chuyển XPASS.
- Biến thể nén khai báo trong `config/parts/index.yaml` **với tên mới** (không sửa `stem` có sẵn):
  ```yaml
  indexes:
    stem_vbyte: {stem: true, remove_stopwords: true, include_title: true, codec: delta_vbyte}
  ```
  rồi `uv run python scripts/build_index.py --index stem_vbyte`.
- Skip pointer (`index/skip.py`, của Việt) đang chạy trên doc_id đã giải mã. Khi có VByte, cần lưu
  (doc_id, byte offset) theo khối — **trao đổi với Việt** trước khi đổi.

### 5.5 Đánh giá & demo — Vinh

- Chủ `eval/metrics.py`: thêm độ đo mới (vd. "cả 2 đoạn vàng trong top-k", recall mức câu từ
  `supporting_facts`) bằng **hàm mới**, giữ nguyên chữ ký `evaluate_per_query`, `aggregate`
  (đang được `sparse_experiments.py` và `bm25_grid.py` dùng).
- Dữ liệu vẽ đồ thị phần truy xuất thưa có sẵn: `runs/sparse/bm25_grid.csv` (heatmap k1×b),
  `champion.csv`, `tiered.csv`, `smart.csv`, `skip.csv`.

## 6. Quy ước code

- Python 3.11, type hint đầy đủ, docstring tiếng Việt ngắn gọn.
- **Không hardcode tham số** — mọi tham số trong config (mục 5.1).
- `logging` thay cho `print`; gọi `setup_logging()` **đầu tiên** trong mỗi script (trước argparse —
  để tiếng Việt không lỗi trên Windows); `tqdm` cho vòng lặp dài.
- Mọi cấu hình xuất **TREC run**: `qid Q0 doc_id rank score run_name`, chấm bằng harness chung.
- Truy vấn luôn tokenize qua `index.analyze(...)` / `tokenize(...)` dùng chung — không tự viết tokenizer.
- Mỗi module mới có ít nhất một unit test; test cần dữ liệu thật thì gắn `@pytest.mark.data`.

## 7. Quy trình git

```bash
git checkout main && git pull
git checkout -b feat/<ten>-<viec>        # vd. feat/khiem-dense-bge
# ... code + test ...
uv run pytest                            # xanh mới đẩy
git push -u origin feat/<ten>-<viec>     # rồi mở Pull Request vào main
```

- Trước khi mở PR: `git pull --rebase origin main` để cập nhật, chạy lại `uv run pytest`.
- **Thêm thư viện**: `sentence-transformers`, `faiss-cpu`, `spacy` đã khai báo sẵn (`--extra dense`,
  `--extra ner`, đã pin version). Cần thư viện khác → một PR riêng chỉ sửa `pyproject.toml` + `uv.lock`
  (`uv add <gói>`), báo nhóm merge sớm — `uv.lock` là file dễ conflict nhất.
- Kết quả thí nghiệm: ghi bảng nhỏ (CSV/JSON/MD) vào `runs/<phần>/` — `.gitignore` đã tự cho phép
  các đuôi này ở mọi `runs/<phần>/` và tự bỏ qua run file TREC, **không cần sửa `.gitignore`**.

Checklist PR:
- [ ] Chỉ sửa file trong vùng của mình (mục 4)
- [ ] `uv run pytest` xanh (kể cả `-m data` trên máy có dữ liệu)
- [ ] Không có file trong `data/`, không có run file TREC
- [ ] Cấu hình mới nằm trong `config/parts/<phần>.yaml`
- [ ] Cập nhật dòng tiến độ của mình trong README
