#!/usr/bin/env python3
"""
Collect negotiation logs from scenario result folders.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set
import shutil


@dataclass(frozen=True)
class CopyResult:
    scenario_dir: Path
    source: Path
    destination: Path
    source_count: int


def iter_scenario_dirs(root: Path, skip_names: Iterable[str]) -> Iterable[Path]:
    """Yield scenario directories directly under root, skipping unwanted names."""
    skip: Set[str] = set(skip_names)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skip:
            continue
        yield child


def collect_nego_logs(
    root: Path,
    destination: Path | None = None,
    overwrite: bool = True,
    skip_names: Sequence[str] = ("collected_nego_logs",),
) -> tuple[List[CopyResult], List[Path], List[CopyResult]]:
    """
    Copy each scenario's nego.json file into a destination directory.

    Returns a tuple of:
      copied results, missing scenario directories, and scenarios with duplicates.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    if destination is None:
        destination = root / "collected_nego_logs"
    else:
        destination = destination.expanduser().resolve()

    destination.mkdir(parents=True, exist_ok=True)

    copied: List[CopyResult] = []
    missing: List[Path] = []
    duplicates: List[CopyResult] = []

    for scenario_dir in iter_scenario_dirs(root, skip_names):
        nego_files = sorted(scenario_dir.glob("*/log/nego.json"))
        if not nego_files:
            missing.append(scenario_dir)
            continue

        source = nego_files[-1]
        target_name = f"{scenario_dir.name}.json"
        target_path = destination / target_name

        if target_path.exists() and not overwrite:
            continue

        shutil.copy2(source, target_path)
        result = CopyResult(
            scenario_dir=scenario_dir,
            source=source,
            destination=target_path,
            source_count=len(nego_files),
        )
        copied.append(result)

        if len(nego_files) > 1:
            duplicates.append(result)

    return copied, missing, duplicates


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect nego.json files from scenario result directories.",
    )
    parser.add_argument(
        "root",
        help="Root directory containing scenario runs (e.g. results/.../image/v2x_final).",
    )
    parser.add_argument(
        "--dest",
        help="Destination directory for collected logs (default: <root>/collected_nego_logs).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing files in the destination directory.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Additional directory names to skip under the root.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    root = Path(args.root)
    destination = Path(args.dest) if args.dest else None
    skip_names: Sequence[str] = ("collected_nego_logs",)
    if args.skip:
        skip_names = tuple(set(skip_names).union(args.skip))

    copied, missing, duplicates = collect_nego_logs(
        root=root,
        destination=destination,
        overwrite=not args.no_overwrite,
        skip_names=skip_names,
    )

    dest_path = (destination or (root.expanduser().resolve() / "collected_nego_logs"))
    print(f"Copied {len(copied)} nego.json files to {dest_path}")

    if missing:
        print("Missing nego.json for scenarios:")
        for path in missing:
            print(f"  {path}")

    if duplicates:
        print("Scenarios with multiple nego.json files (latest copied):")
        for result in duplicates:
            print(f"  {result.scenario_dir}")
            print(f"    kept: {result.source}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
