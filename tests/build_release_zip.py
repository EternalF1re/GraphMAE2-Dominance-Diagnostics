"""Build a deterministic release ZIP with POSIX member paths."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__", "reproduced"}
FIXED_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def iter_release_files(root: Path):
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if EXCLUDED_PARTS.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        yield path, relative


def build_zip(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in iter_release_files(root):
            member = (Path(root.name) / relative).as_posix()
            if "\\" in member:
                raise RuntimeError(f"non-POSIX member path: {member}")
            info = zipfile.ZipInfo(member, FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_zip(args.root, args.output)
    print(f"release ZIP: {args.output.resolve()}")


if __name__ == "__main__":
    main()
