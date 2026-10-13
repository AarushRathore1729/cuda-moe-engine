"""Build a byte-reproducible ZIP of source, recorded evidence, and demos."""

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist",
            "runs", ".ipynb_checkpoints"}


def build(destination):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = "cuda-moe-engine-v0.2.0"
    paths = []
    for p in sorted(ROOT.rglob("*")):
        relative = p.relative_to(ROOT)
        if (not p.is_file() or p.resolve() == destination or any(
            part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts
        )):
            continue
        paths.append((p, relative.as_posix()))
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p, relative in paths:
            entry = zipfile.ZipInfo(f"{prefix}/{relative}", date_time=(2026, 10, 13, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (0o100644 << 16)
            archive.writestr(entry, p.read_bytes(), compresslevel=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n")
    print(f"{destination}\n{len(paths)} files; SHA-256: {digest}")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/cuda-moe-engine-v0.2.0.zip")
    args = parser.parse_args()
    build(args.output)
