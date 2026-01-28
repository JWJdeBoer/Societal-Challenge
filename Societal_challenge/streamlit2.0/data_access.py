# data_access.py
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import streamlit as st


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def load_csv_preview(csv_path: str, nrows: int = 200) -> pd.DataFrame:
    """Fast preview load. Does not parse geometry."""
    return pd.read_csv(csv_path, nrows=nrows)


def save_uploaded_csv(uploaded_file) -> Tuple[str, str]:
    """Persist uploaded CSV to a stable path and return (path, fingerprint)."""
    data = uploaded_file.getvalue()
    fingerprint = _bytes_sha256(data)

    uploads_dir = Path(".streamlit_uploads")
    uploads_dir.mkdir(exist_ok=True)

    out_path = uploads_dir / f"combined_{fingerprint[:12]}.csv"
    if not out_path.exists():
        out_path.write_bytes(data)

    return str(out_path), fingerprint


def file_fingerprint(path: str) -> str:
    """Stable hash for caching. Uses mtime+size (fast) and a short content hash sample."""
    p = Path(path)
    if not p.exists():
        return "missing"
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))
    # sample first 64KB
    with p.open("rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()
