import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, DATASET_NAME
from src.data import get_source_file_paths

CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    data_dir: Path = DATA_DIR,
    dataset_name: str = DATASET_NAME,
) -> dict[str, Any]:
    files = []
    for path in sorted(get_source_file_paths(data_dir), key=lambda item: item.name):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "relative_path": str(path.relative_to(data_dir.parent)),
                "size_bytes": stat.st_size,
                "modified_at_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=UTC,
                ).isoformat(),
                "sha256": file_sha256(path),
            }
        )

    dataset_digest = hashlib.sha256()
    for item in files:
        dataset_digest.update(item["name"].encode())
        dataset_digest.update(item["sha256"].encode())

    version = dataset_digest.hexdigest()
    return {
        "dataset_name": dataset_name,
        "dataset_version": version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
    }


def write_dataset_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path
