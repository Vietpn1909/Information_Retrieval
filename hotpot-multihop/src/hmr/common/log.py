"""Cấu hình logging thống nhất cho mọi script."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Định dạng log; gọi ĐẦU TIÊN trong mỗi script (trước argparse).

    Ép stdout/stderr sang UTF-8: Windows mặc định cp1252 khi output bị chuyển hướng,
    làm ``--help`` và log tiếng Việt bị lỗi UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "urllib3", "huggingface_hub", "datasets", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
