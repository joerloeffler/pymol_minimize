#!/usr/bin/env python
"""Headless batch AIB relaxation without relying on PyMOL plugin autoloading."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pymol import cmd
from pymol_quick_minimize import MinimizationError, quick_relax


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick-relax AIB residues in complete PDB structures."
    )
    parser.add_argument("pdbs", nargs="+", type=Path, help="Input PDB files")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("minimized"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for source in args.pdbs:
        output = args.output_dir / f"{source.stem}_AIB_relaxed.pdb"
        try:
            cmd.reinitialize()
            cmd.load(str(source), "mol")
            atom_count = cmd.count_atoms("mol")
            aib_count = cmd.count_atoms("mol and resn AIB")
            if atom_count == 0:
                raise MinimizationError("PyMOL loaded zero atoms")
            if aib_count == 0:
                raise MinimizationError("structure contains no AIB atoms")
            result = quick_relax("resn AIB", "mol", quiet=1, _cmd=cmd)
            cmd.save(str(output), "mol", state=1)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("output file was not written")
            print(
                f"OK  {source} -> {output} "
                f"({aib_count} displayed AIB atoms; {result.movable_atom_count} prepared atoms movable)"
            )
        except Exception as exc:
            failures.append(source)
            print(f"FAIL  {source}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)}/{len(args.pdbs)} structures failed.", file=sys.stderr)
        return 1
    print(f"\nCompleted {len(args.pdbs)} structures successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
