"""Nạp cấu hình YAML. Mọi tham số (k1, b, top_k, đường dẫn) lấy từ đây.

Cấu hình = ``config/default.yaml`` (phần dùng chung) + mọi file ``config/parts/*.yaml``
(mỗi thành viên một file riêng → không sửa chung một file, tránh conflict khi merge).
Gộp theo khoá cấp 1; với khoá là dict (vd. ``retrievers``, ``paths``) thì gộp thêm một cấp.
Trùng khoá ở bất kỳ cấp nào → báo lỗi ngay, không âm thầm ghi đè.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
PARTS_DIRNAME = "parts"

Config = dict[str, Any]


def merge_part(cfg: Config, part: Mapping[str, Any], source: str) -> None:
    """Gộp một file part vào ``cfg`` (tại chỗ); trùng khoá thì ném ``ValueError``."""
    for key, value in part.items():
        if key not in cfg:
            cfg[key] = value
        elif isinstance(cfg[key], dict) and isinstance(value, dict):
            dup = sorted(cfg[key].keys() & value.keys())
            if dup:
                raise ValueError(f"{source}: khoá {key}.{dup} đã được định nghĩa ở file khác — hãy đổi tên")
            cfg[key].update(value)
        else:
            raise ValueError(f"{source}: khoá '{key}' đã được định nghĩa ở file khác — hãy đổi tên")


def load_config(path: str | Path | None = None) -> Config:
    """Đọc YAML chính + các part; đường dẫn trong ``paths`` quy về tuyệt đối theo gốc repo.

    Gốc repo = thư mục cha của thư mục chứa file config.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        cfg: Config = yaml.safe_load(f)
    for part_path in sorted((cfg_path.parent / PARTS_DIRNAME).glob("*.yaml")):
        with part_path.open(encoding="utf-8") as f:
            merge_part(cfg, yaml.safe_load(f) or {}, part_path.name)
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
