"""Runtime dependency and backend checks with actionable messages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from shutil import which
from typing import Iterable


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    summary: str
    details: tuple[str, ...]


_PACKAGES = (
    ("pymol", "pymol-open-source"),
    ("openmm", "openmm"),
    ("openmmforcefields", "openmmforcefields"),
    ("openff.toolkit", "openff-toolkit"),
    ("rdkit", "rdkit"),
)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown version"


def _lines(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(values)


def check_installation(quiet=0) -> CheckResult:
    """Check imports, OpenMM CPU support, Sage data, and AM1-BCC tooling."""
    details = []
    failures = []
    for module_name, distribution in _PACKAGES:
        try:
            import_module(module_name)
            details.append(f"OK  {distribution} ({_version(distribution)})")
        except Exception as exc:
            failures.append(f"MISSING/BROKEN  {distribution}: {exc}")

    if not failures:
        try:
            from openmm import Platform

            platform = Platform.getPlatformByName("CPU")
            details.append(f"OK  OpenMM platform: {platform.getName()}")
        except Exception as exc:
            failures.append(f"MISSING/BROKEN  OpenMM CPU platform: {exc}")

        try:
            from openmmforcefields.generators import SystemGenerator

            available = set(SystemGenerator.SMALL_MOLECULE_FORCEFIELDS)
            if "openff-2.2.1" not in available:
                raise RuntimeError("OpenFF Sage 2.2.1 data are not installed")
            details.append("OK  OpenFF Sage 2.2.1")
        except Exception as exc:
            failures.append(f"MISSING/BROKEN  force-field data: {exc}")

    amber_programs = {
        "antechamber": "AM1-BCC ligand charges",
        "cpptraj": "glycan recognition/preparation",
        "tleap": "Amber/GLYCAM topology compilation",
    }
    for program, purpose in amber_programs.items():
        executable = which(program)
        if executable:
            details.append(f"OK  AmberTools/{program}: {executable}")
        else:
            failures.append(f"MISSING  AmberTools {program} (needed for {purpose})")

    all_details = _lines((*details, *failures))
    result = CheckResult(
        ok=not failures,
        summary="Quick Minimize is ready." if not failures else "Quick Minimize is not ready.",
        details=all_details,
    )
    if str(quiet).lower() not in {"1", "true", "yes", "on"}:
        print(result.summary)
        print("\n".join(result.details))
        if failures:
            print("Install the complete environment with: conda env create -f environment.yml")
    return result
