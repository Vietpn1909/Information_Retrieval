"""Nạp cấu hình YAML. Mọi tham số (k1, b, top_k, đường dẫn) lấy từ đây."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"

Config = dict[str, Any]


def load_config(path: str | Path | None = None) -> Config:
    """Đọc YAML; các đường dẫn trong ``paths`` được quy về tuyệt đối theo gốc repo.

    Gốc repo = thư mục cha của thư mục chứa file config.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        cfg: Config = yaml.safe_load(f)
    root = cfg_path.resolve().parent.parent
    cfg["paths"] = {k: str((root / v).resolve()) for k, v in cfg.get("paths", {}).items()}
    return cfg


def retriever_config(cfg: Config, name: str) -> dict[str, Any]:
    """Lấy cấu hình truy xuất theo tên (khoá trong ``retrievers``)."""
    retrievers: dict[str, Any] = cfg.get("retrievers", {})
    if name not in retrievers:
        raise KeyError(f"Không có cấu hình '{name}'. Có: {sorted(retrievers)}")
    return retrievers[name]


def index_path(cfg: Config, index_name: str) -> Path:
    """Đường dẫn file pickle của biến thể chỉ mục ``index_name``."""
    return Path(cfg["paths"]["index_dir"]) / f"{index_name}.pkl"
