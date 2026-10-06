"""Robust file download, adapted from Project 1 (dimreduce_agent/datasets.py).

Zenodo, MedMNIST's host, returns 502/504 under load, and a truncated download
is still a valid-looking file. So each URL is retried with exponential backoff,
the payload goes to a ``.partial`` file, and it is moved into place only after
its MD5 matches the published checksum. The checksum is what makes falling back
to a mirror safe.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_with_retries(urls: list[str], dest: Path, expected_md5: str | None = None,
                       attempts_per_url: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_error: Exception | None = None
    for url in urls:
        host = url.split("/")[2]
        for attempt in range(1, attempts_per_url + 1):
            try:
                digest = hashlib.md5()
                with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
                if expected_md5 and digest.hexdigest() != expected_md5:
                    raise ValueError(f"checksum mismatch: expected {expected_md5}, "
                                     f"got {digest.hexdigest()}")
                tmp.replace(dest)
                print(f"    fetched {dest.name} from {host}"
                      f"{' (MD5 verified)' if expected_md5 else ''}")
                return
            except (urllib.error.URLError, ValueError, TimeoutError, ConnectionError) as err:
                tmp.unlink(missing_ok=True)
                last_error = err
                if attempt < attempts_per_url:
                    time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {dest.name}; last error: {last_error}. "
                       f"Tried: {urls}. Save it manually to {dest}")
