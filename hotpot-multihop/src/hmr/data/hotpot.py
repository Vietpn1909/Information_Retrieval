"""Tải HotpotQA từ HuggingFace và lấy mẫu câu hỏi."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Mapping
from typing import Any

logger = logging.getLogger(__name__)

Example = Mapping[str, Any]


def sample_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Lấy mẫu ``n`` chỉ số trong ``[0, n_total)`` cố định theo seed, sắp tăng."""
    if n >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), n))


def load_examples(data_cfg: Mapping[str, Any]) -> Iterator[tuple[str, Example]]:
    """Sinh ``(split, example)``: N câu train lấy mẫu rồi toàn bộ dev.

    split được ghi là ``"train"`` hoặc ``"dev"``.
    """
    from datasets import load_dataset

    ds = load_dataset(data_cfg["hf_name"], data_cfg["hf_config"])
    train = ds["train"]
    idx = sample_indices(len(train), int(data_cfg["n_train"]), int(data_cfg["seed"]))
    logger.info("Lấy %d/%d câu train (seed=%s)", len(idx), len(train), data_cfg["seed"])
    for ex in train.select(idx):
        yield "train", ex
    dev = ds[data_cfg["dev_split"]]
    logger.info("Lấy toàn bộ %d câu dev", len(dev))
    for ex in dev:
        yield "dev", ex
