#!/usr/bin/env python3
"""Validate and safely extract the official Auto-AVSR LRS3 landmark archive."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_COUNTS = {"pretrain": 118_516, "trainval": 31_982, "test": 1_321}


def landmark_counts(root: Path) -> dict[str, int]:
    return {
        split: sum(1 for _ in (root / split).rglob("*.pkl"))
        for split in EXPECTED_COUNTS
    }


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
        counts = landmark_counts(destination)
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(
                f"existing landmark directory has unexpected counts: {counts}; "
                f"expected {EXPECTED_COUNTS}"
            )
        print(f"LRS3_LANDMARKS_ALREADY_EXTRACTED counts={counts}")
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
        counts = landmark_counts(source)
        if counts != EXPECTED_COUNTS:
            raise RuntimeError(
                f"landmark archive has unexpected counts: {counts}; "
                f"expected {EXPECTED_COUNTS}"
            )
        source.replace(destination)
        print(f"LRS3_LANDMARKS_OK counts={counts}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
