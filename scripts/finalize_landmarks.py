#!/usr/bin/env python3
"""Validate and safely extract the official Auto-AVSR LRS3 landmark archive."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "LRS3_landmarks"

    if destination.exists():
        count = sum(1 for _ in destination.rglob("*.pkl"))
        if count < 151_000:
            raise RuntimeError(f"existing landmark directory is incomplete: {count}")
        print(f"LRS3_LANDMARKS_ALREADY_EXTRACTED count={count}")
        return

    temporary = Path(tempfile.mkdtemp(prefix=".lrs3-landmarks-", dir=output_root))
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"unsafe archive member: {member.filename}")
            bad_member = bundle.testzip()
            if bad_member is not None:
                raise RuntimeError(f"CRC failure in archive member: {bad_member}")
            bundle.extractall(temporary)

        candidates = []
        for path in temporary.rglob("*"):
            if not path.is_dir():
                continue
            if all((path / split).is_dir() for split in ("pretrain", "trainval", "test")):
                candidates.append(path)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one landmark root, got {candidates}")
        source = candidates[0]
        counts = {
            split: sum(1 for _ in (source / split).rglob("*.pkl"))
            for split in ("pretrain", "trainval", "test")
        }
        if sum(counts.values()) < 151_000:
            raise RuntimeError(f"landmark archive is incomplete: {counts}")
        source.replace(destination)
        print(f"LRS3_LANDMARKS_OK counts={counts}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
