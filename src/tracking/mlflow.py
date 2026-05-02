import mlflow


def log_dataset_version(manifest: dict, manifest_path: object) -> None:
    source_files = ",".join(item["name"] for item in manifest["files"])
    mlflow.set_tags(
        {
            "dataset.name": manifest["dataset_name"],
            "dataset.version": manifest["dataset_version"],
            "dataset.hash": manifest["dataset_version"],
            "dataset.source_files": source_files,
        }
    )
    mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
