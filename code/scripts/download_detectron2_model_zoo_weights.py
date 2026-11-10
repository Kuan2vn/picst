#!/usr/bin/env python
"""Download Detectron2 model-zoo weights used by USIS10K instance baselines.

Run this script on a machine or login node with internet access, then copy the
downloaded .pkl file to the offline training node and pass it with --weights or
BASE_WEIGHTS.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


STATIC_WEIGHTS: Dict[str, Tuple[str, str]] = {
    "mask_rcnn": (
        "model_final_f10217.pkl",
        "https://dl.fbaipublicfiles.com/detectron2/"
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/"
        "137849600/model_final_f10217.pkl",
    ),
    "cascade_mask_rcnn": (
        "model_final_480dd8.pkl",
        "https://dl.fbaipublicfiles.com/detectron2/"
        "Misc/cascade_mask_rcnn_R_50_FPN_3x/"
        "144998488/model_final_480dd8.pkl",
    ),
}

BACKBONE_CONFIGS = {
    "mask_rcnn": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "cascade_mask_rcnn": "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
    "pointrend_mask_rcnn": "PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONE_CONFIGS) + ("all",),
        default="mask_rcnn",
    )
    parser.add_argument("--output-dir", default="pretrain/detectron2")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print URLs and target paths without downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload even if the target file already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Socket timeout in seconds for each network operation.",
    )
    parser.add_argument(
        "--chunk-mb",
        type=float,
        default=4.0,
        help="Download chunk size in MiB.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress messages.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing .tmp partial download.",
    )
    return parser.parse_args()


def resolve_with_detectron2(backbone: str) -> Optional[Tuple[str, str]]:
    try:
        from detectron2.model_zoo import get_checkpoint_url
    except Exception:
        return None
    url = get_checkpoint_url(BACKBONE_CONFIGS[backbone])
    return Path(url).name, url


def resolve_weight(backbone: str) -> Tuple[str, str]:
    if backbone in STATIC_WEIGHTS:
        return STATIC_WEIGHTS[backbone]
    resolved = resolve_with_detectron2(backbone)
    if resolved is None:
        raise SystemExit(
            f"No static URL is recorded for {backbone}. Install Detectron2 and "
            "rerun this script on an internet-enabled machine, or use "
            "mask_rcnn/cascade_mask_rcnn."
        )
    return resolved


def iter_backbones(choice: str) -> Iterable[str]:
    if choice == "all":
        return BACKBONE_CONFIGS.keys()
    return (choice,)


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "?"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024.0:
            return f"{amount:.1f}{unit}"
        amount /= 1024.0
    return f"{amount:.1f}TiB"


def download(
    url: str,
    target: Path,
    overwrite: bool,
    timeout: float,
    chunk_size: int,
    progress_interval: float,
    resume: bool,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        print(f"exists: {target}")
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    if overwrite and tmp.exists():
        tmp.unlink()

    existing = tmp.stat().st_size if resume and tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    mode = "ab" if existing > 0 else "wb"

    print(f"download: {url}")
    print(f"      to: {target}")
    if existing:
        print(f"  resume: {format_bytes(existing)} already downloaded in {tmp}")

    request = urllib.request.Request(url, headers=headers)
    started_at = time.monotonic()
    last_report = started_at
    downloaded = existing

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if existing and getattr(response, "status", 200) == 200:
                print("  server did not accept resume; restarting from 0")
                existing = 0
                downloaded = 0
                mode = "wb"
            content_length = response.headers.get("Content-Length")
            total = int(content_length) + existing if content_length else None

            with tmp.open(mode) as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= progress_interval:
                        elapsed = max(now - started_at, 1e-6)
                        speed = (downloaded - existing) / elapsed
                        print(
                            "progress: "
                            f"{format_bytes(downloaded)}/"
                            f"{format_bytes(total)} "
                            f"({format_bytes(int(speed))}/s)",
                            flush=True,
                        )
                        last_report = now
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Download failed: {exc}. If this is an offline HPC node, run this "
            "script on a machine with internet and copy the .pkl file back."
        ) from exc

    size = tmp.stat().st_size
    print(f"complete: {format_bytes(size)}")
    tmp.replace(target)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)

    for backbone in iter_backbones(args.backbone):
        filename, url = resolve_weight(backbone)
        target = out_root / backbone / filename
        print(f"{backbone}.url={url}")
        print(f"{backbone}.path={target}")
        print(f"{backbone}.train_arg=--weights {target}")
        if not args.print_only:
            download(
                url,
                target,
                args.overwrite,
                args.timeout,
                max(1, int(args.chunk_mb * 1024 * 1024)),
                args.progress_interval,
                not args.no_resume,
            )

    if args.print_only:
        return
    print("done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
