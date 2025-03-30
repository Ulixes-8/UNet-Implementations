"""
Script: download_and_extract.py

Short Description:
    Downloads a .tar file from Google Drive and extracts it into the repo-level data/raw/ folder.

Description:
    This script creates data/raw/ and data/processed/ folders in the root of the repository if they don't exist.
    It then downloads a .tar file from Google Drive using its file ID and extracts it into data/raw/.

Example Usage:
    python scripts/download_and_extract.py
"""

import tarfile
from pathlib import Path
import gdown


def create_data_dirs(repo_root: Path) -> tuple[Path, Path]:
    """
    Creates the `data/raw/` and `data/processed/` directories if they don't already exist.

    Args:
        repo_root (Path): The root directory of the repository.

    Returns:
        Tuple[Path, Path]: Paths to raw/ and processed/ directories.
    """
    data_dir = repo_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, processed_dir


def download_from_drive(file_id: str, output_path: Path) -> None:
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        file_id (str): Google Drive file ID from the shareable link.
        output_path (Path): Full path where the file will be saved.

    Returns:
        None
    """
    print(f"Downloading file from Google Drive ID: {file_id}")
    gdown.download(id=file_id, output=str(output_path), quiet=False)
    print(f"File downloaded to: {output_path}")


def extract_tar_file(tar_path: Path, extract_to: Path) -> None:
    """
    Extracts a .tar archive into the given directory.

    Args:
        tar_path (Path): Path to the .tar file.
        extract_to (Path): Directory to extract contents into.

    Returns:
        None
    """
    print(f"Extracting {tar_path.name} to {extract_to}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")


def main() -> None:
    """
    Main function to download and extract dataset into data/raw/.

    Returns:
        None
    """
    # Google Drive file ID
    file_id: str = "1-BMGfI4_XfwDk5SZL6JoDSLxt7rYiAeI"
    tar_filename: str = "Data_filtered.tar"

    # Get absolute path to the repo root
    repo_root: Path = Path(__file__).resolve().parents[1]

    # Create required directories
    raw_dir, _ = create_data_dirs(repo_root)

    # Define path to download the tar file
    tar_path: Path = raw_dir / tar_filename

    # Download and extract
    download_from_drive(file_id, tar_path)
    extract_tar_file(tar_path, raw_dir)


if __name__ == "__main__":
    main()
