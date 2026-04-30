from pathlib import Path

from src.config import RAW_DATA_FILES
from src.dataset_version import build_dataset_manifest


def write_raw_files(data_dir: Path, suffix: str = "") -> None:
    data_dir.mkdir(exist_ok=True)
    for file_name in RAW_DATA_FILES.values():
        (data_dir / file_name).write_bytes(f"{file_name}{suffix}".encode())


def test_dataset_version_is_stable_for_same_files(tmp_path: Path) -> None:
    write_raw_files(tmp_path)

    first_manifest = build_dataset_manifest(tmp_path)
    second_manifest = build_dataset_manifest(tmp_path)

    assert first_manifest["dataset_version"] == second_manifest["dataset_version"]


def test_dataset_version_changes_when_source_file_changes(tmp_path: Path) -> None:
    write_raw_files(tmp_path)
    first_manifest = build_dataset_manifest(tmp_path)

    changed_file = tmp_path / RAW_DATA_FILES["status"]
    changed_file.write_bytes(b"changed")
    second_manifest = build_dataset_manifest(tmp_path)

    assert first_manifest["dataset_version"] != second_manifest["dataset_version"]
