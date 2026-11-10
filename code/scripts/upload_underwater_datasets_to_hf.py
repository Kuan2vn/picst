#!/usr/bin/env python3
"""Upload the prepared UIIS and USIS10K datasets to Hugging Face Hub.

The uploader uses explicit allow-lists. It never uploads papers, source code,
debug folders, Python caches, logs, or any other files outside the layouts
listed below.

Default UIIS content:
    train_val/images/*.jpg
    train_val/masks/*.png
    TEST/images/*.jpg
    TEST/masks/*.png
    metadata/categories.json              (when present)
    original_instance_annotations/*.json  (fetched from LiamLian0727/UIIS)

Default USIS10K content:
    USIS10K/{train,val,test}/*.jpg
    USIS10K/multi_class_annotations/*.json
    USIS10K/foreground_annotations/*.json
    semantic_{bg,ignore}/{train,val,test}/masks/*.png
    semantic_{bg,ignore}/metadata/categories.json  (when present)

Examples:
    # Inspect the exact manifest without creating or modifying Hub repos.
    python scripts/upload_underwater_datasets_to_hf.py \
        --uiis-repo-id USER/UIIS-semantic \
        --usis-repo-id USER/USIS10K-semantic \
        --dry-run

    # Upload to two private dataset repositories (the safe default).
    python scripts/upload_underwater_datasets_to_hf.py \
        --uiis-repo-id USER/UIIS-semantic \
        --usis-repo-id USER/USIS10K-semantic \
        --upload

    # Upload both UIIS mask encodings, or make newly created repos public.
    python scripts/upload_underwater_datasets_to_hf.py \
        --uiis-repo-id USER/UIIS-semantic \
        --usis-repo-id USER/USIS10K-semantic \
        --uiis-mask-format both --public --upload

Install/update the Hub client first on the HPC:
    python -m pip install -U "huggingface_hub[hf_xet]"

Authentication is read from the normal Hugging Face login/cache. Tokens are
never accepted as command-line arguments, so they cannot leak into shell logs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
UIIS_UPSTREAM_REPO = "LiamLian0727/UIIS"
UIIS_EXPECTED_INSTANCE_JSON = ("train.json", "val.json")
USIS_SPLITS = ("train", "val", "test")
UIIS_SPLITS = ("train_val", "TEST")
# Hugging Face rejects a single commit containing more than 25k large-file
# operations. Keep margin below that hard limit and force the resumable
# multi-commit uploader for larger manifests, independently of hf_xet state.
SINGLE_COMMIT_FILE_LIMIT = 24_000


@dataclass
class UploadPlan:
    name: str
    repo_id: str
    root: Path
    allow_patterns: List[str]
    files: List[Path]
    extra_folder: Optional[Path] = None
    extra_path_in_repo: Optional[str] = None
    extra_patterns: Optional[List[str]] = None
    extra_files: Optional[List[Path]] = None

    @property
    def byte_count(self) -> int:
        main_bytes = sum(path.stat().st_size for path in self.files)
        extra_bytes = sum(path.stat().st_size for path in (self.extra_files or []))
        return main_bytes + extra_bytes

    @property
    def file_count(self) -> int:
        return len(self.files) + len(self.extra_files or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload only the required UIIS/USIS10K data and masks to Hugging Face."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SND_ReMask repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--uiis-root",
        type=Path,
        default=None,
        help="Prepared UIIS root (default: PROJECT/data/UIIS/orin_data).",
    )
    parser.add_argument(
        "--usis-root",
        type=Path,
        default=None,
        help="USIS10K data root (default: PROJECT/data/USIS10K).",
    )
    parser.add_argument("--uiis-repo-id", help="Destination dataset repo, e.g. USER/UIIS-semantic.")
    parser.add_argument(
        "--usis-repo-id", help="Destination dataset repo, e.g. USER/USIS10K-semantic."
    )
    parser.add_argument(
        "--only",
        choices=("both", "uiis", "usis10k"),
        default="both",
        help="Upload both datasets or only one of them (default: both).",
    )
    parser.add_argument(
        "--uiis-instance-annotations-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing original UIIS train.json and val.json. If omitted, "
            "the uploader fetches only annotations/*.json from LiamLian0727/UIIS."
        ),
    )
    parser.add_argument(
        "--skip-uiis-original-annotations",
        action="store_true",
        help="Do not include the original UIIS COCO instance annotations.",
    )
    parser.add_argument(
        "--uiis-mask-format",
        choices=("png", "bmp", "both"),
        default="png",
        help=(
            "UIIS semantic-mask encoding to upload (default: png). Indexed PNG is "
            "lossless and dramatically smaller; BMP is retained only for legacy compatibility."
        ),
    )
    parser.add_argument(
        "--include-prepare-reports",
        action="store_true",
        help="Include prepare_report.json files, which can contain local HPC paths.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public",
        action="store_true",
        help="Create missing dataset repos as public. The default is private.",
    )
    visibility.add_argument(
        "--private",
        action="store_true",
        help="Explicitly request the default private visibility for newly created repos.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local data and print the upload manifest; do not contact the Hub.",
    )
    action.add_argument(
        "--upload",
        action="store_true",
        help="Create missing repos, upload the allow-listed files, and verify remote paths.",
    )
    return parser.parse_args()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return "{:.2f} {}".format(size, unit)
        size /= 1024.0
    raise AssertionError("unreachable")


def list_direct_files(folder: Path, suffixes: Sequence[str]) -> List[Path]:
    accepted = {suffix.lower() for suffix in suffixes}
    if not folder.is_dir():
        raise FileNotFoundError("Missing directory: {}".format(folder))
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in accepted
    )


def validate_pairs(
    dataset: str,
    split: str,
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str,
) -> Tuple[List[Path], List[Path]]:
    images = list_direct_files(image_dir, IMAGE_SUFFIXES)
    masks = list_direct_files(mask_dir, (mask_suffix,))
    if not images:
        raise RuntimeError("{} {} contains no images: {}".format(dataset, split, image_dir))
    if not masks:
        raise RuntimeError("{} {} contains no {} masks: {}".format(dataset, split, mask_suffix, mask_dir))

    image_stems = {path.stem for path in images}
    mask_stems = {path.stem for path in masks}
    missing_masks = sorted(image_stems - mask_stems)
    orphan_masks = sorted(mask_stems - image_stems)
    if missing_masks or orphan_masks:
        details = []
        if missing_masks:
            details.append(
                "missing masks={} (examples: {})".format(
                    len(missing_masks), ", ".join(missing_masks[:5])
                )
            )
        if orphan_masks:
            details.append(
                "orphan masks={} (examples: {})".format(
                    len(orphan_masks), ", ".join(orphan_masks[:5])
                )
            )
        raise RuntimeError("{} {} pairing failed: {}".format(dataset, split, "; ".join(details)))
    return images, masks


def unique_sorted(paths: Iterable[Path]) -> List[Path]:
    return sorted(set(paths), key=lambda path: path.as_posix())


def files_matching(root: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return unique_sorted(files)


def require_files(paths: Sequence[Path], description: str) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing {}:\n  {}".format(description, "\n  ".join(missing)))


def build_uiis_plan(args: argparse.Namespace, annotations_dir: Optional[Path]) -> UploadPlan:
    root = (args.uiis_root or args.project_root / "data" / "UIIS" / "orin_data").resolve()
    if not root.is_dir():
        raise FileNotFoundError("Missing UIIS root: {}".format(root))

    patterns: List[str] = []
    validated: List[Path] = []
    for split in UIIS_SPLITS:
        primary_suffix = ".png" if args.uiis_mask_format in ("png", "both") else ".bmp"
        images, masks = validate_pairs(
            "UIIS", split, root / split / "images", root / split / "masks", primary_suffix
        )
        validated.extend(images)
        validated.extend(masks)
        patterns.extend(
            ("{}/images/*.jpg".format(split), "{}/masks/*{}".format(split, primary_suffix))
        )

        if args.uiis_mask_format == "both":
            secondary_suffix = ".bmp"
            secondary_masks = list_direct_files(root / split / "masks", (secondary_suffix,))
            image_stems = {path.stem for path in images}
            secondary_stems = {path.stem for path in secondary_masks}
            if image_stems != secondary_stems:
                raise RuntimeError(
                    "UIIS {} {} masks do not pair exactly with the images".format(
                        split, secondary_suffix
                    )
                )
            validated.extend(secondary_masks)
            patterns.append("{}/masks/*{}".format(split, secondary_suffix))

    categories = root / "metadata" / "categories.json"
    if categories.is_file():
        validated.append(categories)
        patterns.append("metadata/categories.json")
    if args.include_prepare_reports:
        report = root / "metadata" / "prepare_report.json"
        require_files((report,), "UIIS prepare report")
        validated.append(report)
        patterns.append("metadata/prepare_report.json")

    extra_files: List[Path] = []
    if annotations_dir is not None:
        extra_files = [annotations_dir / name for name in UIIS_EXPECTED_INSTANCE_JSON]
        require_files(extra_files, "UIIS original instance annotations")

    return UploadPlan(
        name="UIIS",
        repo_id=args.uiis_repo_id,
        root=root,
        allow_patterns=patterns,
        files=unique_sorted(validated),
        extra_folder=annotations_dir,
        extra_path_in_repo="original_instance_annotations" if annotations_dir else None,
        extra_patterns=list(UIIS_EXPECTED_INSTANCE_JSON) if annotations_dir else None,
        extra_files=extra_files,
    )


def build_usis_plan(args: argparse.Namespace) -> UploadPlan:
    root = (args.usis_root or args.project_root / "data" / "USIS10K").resolve()
    if not root.is_dir():
        raise FileNotFoundError("Missing USIS10K root: {}".format(root))

    raw_root = root / "USIS10K"
    patterns: List[str] = []
    validated: List[Path] = []
    image_stems_by_split: Dict[str, Set[str]] = {}

    for split in USIS_SPLITS:
        images = list_direct_files(raw_root / split, IMAGE_SUFFIXES)
        if not images:
            raise RuntimeError("USIS10K {} contains no images".format(split))
        image_stems_by_split[split] = {path.stem for path in images}
        validated.extend(images)
        patterns.append("USIS10K/{}/*.jpg".format(split))

    annotation_files = [
        raw_root
        / "multi_class_annotations"
        / "multi_class_{}_annotations.json".format(split)
        for split in USIS_SPLITS
    ]
    require_files(annotation_files, "USIS10K multi-class instance annotations")
    validated.extend(annotation_files)
    patterns.append("USIS10K/multi_class_annotations/*.json")

    foreground_dir = raw_root / "foreground_annotations"
    if foreground_dir.is_dir():
        foreground_files = list_direct_files(foreground_dir, (".json",))
        if foreground_files:
            validated.extend(foreground_files)
            patterns.append("USIS10K/foreground_annotations/*.json")

    for mode in ("semantic_bg", "semantic_ignore"):
        for split in USIS_SPLITS:
            mask_dir = root / mode / split / "masks"
            masks = list_direct_files(mask_dir, (".png",))
            mask_stems = {path.stem for path in masks}
            image_stems = image_stems_by_split[split]
            missing_masks = sorted(image_stems - mask_stems)
            orphan_masks = sorted(mask_stems - image_stems)
            if missing_masks or orphan_masks:
                raise RuntimeError(
                    "USIS10K {}/{} pairing failed: {} missing, {} orphan masks".format(
                        mode, split, len(missing_masks), len(orphan_masks)
                    )
                )
            validated.extend(masks)
            patterns.append("{}/{}/masks/*.png".format(mode, split))

        categories = root / mode / "metadata" / "categories.json"
        if categories.is_file():
            validated.append(categories)
            patterns.append("{}/metadata/categories.json".format(mode))
        if args.include_prepare_reports:
            report = root / mode / "metadata" / "prepare_report.json"
            require_files((report,), "{} prepare report".format(mode))
            validated.append(report)
            patterns.append("{}/metadata/prepare_report.json".format(mode))

    return UploadPlan(
        name="USIS10K",
        repo_id=args.usis_repo_id,
        root=root,
        allow_patterns=patterns,
        files=unique_sorted(validated),
    )


def find_local_uiis_annotations(args: argparse.Namespace) -> Optional[Path]:
    if args.skip_uiis_original_annotations:
        return None
    if args.uiis_instance_annotations_dir is not None:
        folder = args.uiis_instance_annotations_dir.expanduser().resolve()
        require_files(
            [folder / name for name in UIIS_EXPECTED_INSTANCE_JSON],
            "UIIS original instance annotations",
        )
        return folder

    candidates = (
        args.project_root / "data" / "UIIS" / "annotations",
        args.project_root / "data" / "UIIS" / "raw" / "annotations",
        args.project_root / "data" / "UIIS" / "original_instance_annotations",
    )
    for folder in candidates:
        if all((folder / name).is_file() for name in UIIS_EXPECTED_INSTANCE_JSON):
            return folder.resolve()
    return None


def fetch_uiis_annotations() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'Missing huggingface_hub. Install it with: python -m pip install -U "huggingface_hub[hf_xet]"'
        ) from exc

    snapshot = Path(
        snapshot_download(
            repo_id=UIIS_UPSTREAM_REPO,
            repo_type="dataset",
            allow_patterns=["annotations/*.json"],
        )
    )
    annotations_dir = snapshot / "annotations"
    require_files(
        [annotations_dir / name for name in UIIS_EXPECTED_INSTANCE_JSON],
        "downloaded UIIS original instance annotations",
    )
    return annotations_dir


def print_plan(plan: UploadPlan) -> None:
    print("\n{} -> {}".format(plan.name, plan.repo_id))
    print("  local root : {}".format(plan.root))
    print("  files      : {:,}".format(plan.file_count))
    print("  size       : {}".format(human_bytes(plan.byte_count)))
    print("  allow-list :")
    for pattern in plan.allow_patterns:
        print("    - {}".format(pattern))
    if plan.extra_folder is not None:
        print("  extra root : {}".format(plan.extra_folder))
        print("  extra dest : {}/".format(plan.extra_path_in_repo))
        for pattern in plan.extra_patterns or []:
            print("    - {}".format(pattern))


def expected_remote_paths(plan: UploadPlan) -> Set[str]:
    expected = {path.relative_to(plan.root).as_posix() for path in plan.files}
    if plan.extra_folder is not None and plan.extra_path_in_repo:
        prefix = plan.extra_path_in_repo.strip("/")
        for path in plan.extra_files or []:
            expected.add("{}/{}".format(prefix, path.relative_to(plan.extra_folder).as_posix()))
    return expected


def has_hf_xet() -> bool:
    """Return whether the streamed Xet upload backend is installed."""
    return importlib.util.find_spec("hf_xet") is not None


def upload_main_folder(api: object, plan: UploadPlan) -> None:
    """Use the best available resilient uploader for a large dataset folder."""
    large_folder_upload = getattr(api, "upload_large_folder", None)

    if plan.file_count > SINGLE_COMMIT_FILE_LIMIT:
        if large_folder_upload is None:
            raise RuntimeError(
                "{} has {:,} files, which exceeds the safe single-commit limit, but "
                "this huggingface_hub version has no upload_large_folder method. "
                "Install/update with: python -m pip install -U huggingface_hub hf_xet".format(
                    plan.name, plan.file_count
                )
            )
        print(
            "Upload backend: upload_large_folder (forced multi-commit mode because "
            "the {:,}-file manifest exceeds the {:,}-file safe commit limit)".format(
                plan.file_count, SINGLE_COMMIT_FILE_LIMIT
            )
        )
        print(
            "NOTE: this backend may create .cache/.huggingface under the local dataset root; "
            "the explicit allow-list prevents that cache from being uploaded."
        )
        large_folder_upload(
            folder_path=str(plan.root),
            repo_id=plan.repo_id,
            repo_type="dataset",
            allow_patterns=plan.allow_patterns,
        )
        return

    if has_hf_xet():
        print(
            "Upload backend: upload_folder + hf_xet "
            "(streamed, chunk-deduplicated, multi-commit, resumable)"
        )
        api.upload_folder(
            folder_path=str(plan.root),
            repo_id=plan.repo_id,
            repo_type="dataset",
            allow_patterns=plan.allow_patterns,
            commit_message="Upload prepared {} data and semantic masks".format(plan.name),
        )
        return

    if large_folder_upload is None:
        raise RuntimeError(
            "hf_xet is not installed and this huggingface_hub version has no "
            "upload_large_folder fallback. Install/update with: "
            'python -m pip install -U "huggingface_hub[hf_xet]"'
        )

    print(
        "Upload backend: upload_large_folder "
        "(legacy-compatible resumable fallback because hf_xet is unavailable)"
    )
    print(
        "NOTE: this backend may create .cache/.huggingface under the local dataset root; "
        "the explicit allow-list prevents that cache from being uploaded."
    )
    large_folder_upload(
        folder_path=str(plan.root),
        repo_id=plan.repo_id,
        repo_type="dataset",
        allow_patterns=plan.allow_patterns,
    )


def upload_plan(api: object, plan: UploadPlan, public: bool) -> None:
    print("\nCreating/checking {} dataset repo: {}".format(plan.name, plan.repo_id))
    api.create_repo(
        repo_id=plan.repo_id,
        repo_type="dataset",
        private=not public,
        exist_ok=True,
    )
    print("Uploading allow-listed {} files...".format(plan.name))
    upload_main_folder(api, plan)

    if plan.extra_folder is not None:
        print("Uploading original UIIS instance annotations...")
        api.upload_folder(
            folder_path=str(plan.extra_folder),
            path_in_repo=plan.extra_path_in_repo,
            repo_id=plan.repo_id,
            repo_type="dataset",
            allow_patterns=plan.extra_patterns,
            commit_message="Add original UIIS COCO instance annotations",
        )

    print("Verifying remote manifest...")
    remote = set(api.list_repo_files(repo_id=plan.repo_id, repo_type="dataset"))
    expected = expected_remote_paths(plan)
    missing = sorted(expected - remote)
    if missing:
        raise RuntimeError(
            "Remote verification failed for {}: {} expected files are missing (examples: {})".format(
                plan.repo_id, len(missing), ", ".join(missing[:10])
            )
        )

    tolerated_hub_files = {".gitattributes", "README.md"}
    extras = sorted(remote - expected - tolerated_hub_files)
    print("Verified {:,} expected files in {}".format(len(expected), plan.repo_id))
    if extras:
        print(
            "WARNING: the existing remote repo already contains {:,} non-manifest files; "
            "this script did not upload or delete them.".format(len(extras)),
            file=sys.stderr,
        )
        print("  examples: {}".format(", ".join(extras[:10])), file=sys.stderr)


def validate_repo_ids(args: argparse.Namespace) -> None:
    if args.only in ("both", "uiis") and not args.uiis_repo_id:
        raise ValueError("--uiis-repo-id is required when uploading/validating UIIS")
    if args.only in ("both", "usis10k") and not args.usis_repo_id:
        raise ValueError("--usis-repo-id is required when uploading/validating USIS10K")


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    validate_repo_ids(args)

    plans: List[UploadPlan] = []
    local_uiis_annotations: Optional[Path] = None
    if args.only in ("both", "uiis"):
        local_uiis_annotations = find_local_uiis_annotations(args)
        if args.upload and local_uiis_annotations is None and not args.skip_uiis_original_annotations:
            print(
                "Original UIIS annotations are not in the project tree; fetching only "
                "annotations/*.json from {}...".format(UIIS_UPSTREAM_REPO)
            )
            local_uiis_annotations = fetch_uiis_annotations()
        plans.append(build_uiis_plan(args, local_uiis_annotations))

    if args.only in ("both", "usis10k"):
        plans.append(build_usis_plan(args))

    for plan in plans:
        print_plan(plan)

    if args.dry_run:
        if (
            args.only in ("both", "uiis")
            and local_uiis_annotations is None
            and not args.skip_uiis_original_annotations
        ):
            print(
                "\nNOTE: during --upload, train.json and val.json will be fetched from {} "
                "and added under original_instance_annotations/.".format(UIIS_UPSTREAM_REPO)
            )
        print("\nDry run complete. No Hub repository was created or modified.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            'Missing huggingface_hub. Install it with: python -m pip install -U "huggingface_hub[hf_xet]"'
        ) from exc

    api = HfApi()
    identity = api.whoami()
    print("\nAuthenticated on Hugging Face as: {}".format(identity.get("name", "unknown")))
    for plan in plans:
        upload_plan(api, plan, public=args.public)

    print("\nAll requested uploads completed and passed remote manifest verification.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
