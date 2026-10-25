import os

"""Fetch and verify the pinned Bollywood subset, preserving source metadata."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/datasets/beastLucifer/navrasa-5000-dataset/resolve/8056599a540110a31adab3cd297134e099aefc65/"
SHA = "7a7dca3a92d6cdef56054baf50a1a6778eae9d609dee8d184c41bd5a626353b3"


def main():
    for folder in ["data/raw", "data/manifests"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "master_manifest.json"]:
        urllib.request.urlretrieve(
            BASE + name, ROOT / "data/manifests" / ("source_" + name)
        )
    archive = ROOT / "data/raw/Bollywood.zip"
    if not archive.exists():
        temp = archive.with_suffix(".partial")
        urllib.request.urlretrieve(BASE + "India/Bollywood.zip", temp)
        temp.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        names = set()
        for item in z.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in {
                ".wav",
                ".mp3",
                ".flac",
            }:
                continue
            name = Path(item.filename).name
            if name in names:
                raise ValueError("Duplicate archive basename")
            names.add(name)
            with z.open(item) as src, (ROOT / "data/raw" / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Imported {len(names)} verified archive audio entries")


if __name__ == "__main__":
    main()
