"""Generate the release-relative, self-excluding SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__", "reproduced"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    for path in root.rglob("*"):
        if not path.is_file() or EXCLUDED_PARTS.intersection(path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "MANIFEST.sha256":
            continue
        entries.append((relative, digest(path)))
    lines = [f"{file_hash}  {relative}" for relative, file_hash in sorted(entries)]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifest entries: {len(lines)}")


if __name__ == "__main__":
    main()
