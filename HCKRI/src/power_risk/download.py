from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .config import DataConfig


def ensure_ton_iot_downloaded(config: DataConfig) -> bool:
    """Download a compact TON_IoT CSV subset when the local data directory is empty."""

    data_dir = Path(config.data_dir)
    existing = sorted(data_dir.rglob(config.file_glob)) if data_dir.exists() else []
    if existing or not config.auto_download:
        return False

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / config.download_filename
    if not target.exists():
        _download_file(config.download_url, target, timeout=config.download_timeout)

    if config.download_sha256:
        _verify_sha256(target, config.download_sha256)

    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target) as archive:
            archive.extractall(data_dir)
    elif target.suffix.lower() in {".tar", ".gz", ".tgz"}:
        with tarfile.open(target) as archive:
            archive.extractall(data_dir)
    return True


def _download_file(url: str, target: Path, timeout: int) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PowerRisk-TON-IoT/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with target.open("wb") as f:
            shutil.copyfileobj(response, f)


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {path}. Expected {expected}, got {actual}."
        )
