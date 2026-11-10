"""Small resumable HTTP byte-range downloader for multi-gigabyte archives."""

from __future__ import annotations

import concurrent.futures
import json
import math
import threading
import time
import urllib.request
from pathlib import Path


def download_ranges(
    opener,
    url: str,
    target: Path,
    size: int,
    *,
    base_offset: int = 0,
    workers: int = 12,
    chunk_size: int = 16 * 1024 * 1024,
) -> None:
    """Download a contiguous segment, resuming only completed, checked chunks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    journal = target.with_name(target.name + ".parts.json")
    count = math.ceil(size / chunk_size)
    complete: set[int] = set()
    if journal.exists():
        state = json.loads(journal.read_text(encoding="utf-8"))
        if state.get("size") != size or state.get("base_offset") != base_offset or state.get("chunk_size") != chunk_size:
            raise RuntimeError(f"Stale range journal: {journal}")
        complete = {int(i) for i in state["complete"] if 0 <= int(i) < count}
    if not partial.exists():
        partial.touch()
    with partial.open("r+b") as stream:
        stream.truncate(size)
    if not journal.exists():
        journal.write_text(json.dumps({"size": size, "base_offset": base_offset, "chunk_size": chunk_size, "complete": []}), encoding="utf-8")

    lock = threading.Lock()
    done_bytes = sum(min(chunk_size, size - i * chunk_size) for i in complete)
    last_report = time.monotonic()

    def fetch(index: int) -> None:
        nonlocal done_bytes, last_report
        start = index * chunk_size
        length = min(chunk_size, size - start)
        first = base_offset + start
        last = first + length - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
                with opener.open(req, timeout=180) as response:
                    if response.status != 206:
                        raise RuntimeError(f"Expected HTTP 206, got {response.status}")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {first}-{last}/"):
                        raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                    with partial.open("r+b") as output:
                        output.seek(start)
                        received = 0
                        while received < length:
                            block = response.read(min(8 * 1024 * 1024, length - received))
                            if not block:
                                raise EOFError(f"Short range response: {received}/{length} bytes")
                            output.write(block)
                            received += len(block)
                with lock:
                    complete.add(index)
                    done_bytes += length
                    temp = journal.with_name(journal.name + ".tmp")
                    temp.write_text(
                        json.dumps({"size": size, "base_offset": base_offset, "chunk_size": chunk_size, "complete": sorted(complete)}),
                        encoding="utf-8",
                    )
                    temp.replace(journal)
                    if time.monotonic() - last_report >= 20 or done_bytes == size:
                        print(f"{target.name}: {done_bytes / 2**30:.2f}/{size / 2**30:.2f} GiB", flush=True)
                        last_report = time.monotonic()
                return
            except (OSError, EOFError, RuntimeError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Failed chunk {index} of {target.name}: {exc}") from exc
                time.sleep(min(30, 2 ** attempt))

    remaining = [i for i in range(count) if i not in complete]
    print(f"{target.name}: {len(complete)}/{count} chunks already complete; {workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, i) for i in remaining]):
            future.result()
    if partial.stat().st_size != size or len(complete) != count:
        raise RuntimeError(f"Incomplete ranged download: {target}")
