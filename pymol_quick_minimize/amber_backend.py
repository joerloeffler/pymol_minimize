"""AmberTools topology compilation followed by OpenMM relaxation.

This backend is used for covalent glycoproteins and supported protein mimetics
that OpenMM's residue-template matcher cannot assemble directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from .core import MinimizationError, MinimizationResult, _as_bool


_PDB_SUGARS = {
    "NAG", "MAN", "BMA", "GLC", "GAL", "FUC", "SIA", "NDG", "BGC",
}

_STANDARD_PROTEIN = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYX", "GLN", "GLU", "GLH",
    "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL", "AIB", "NLN",
}

_AIB_HEAVY_BONDS = {
    frozenset(("N", "CA")), frozenset(("CA", "C")),
    frozenset(("C", "O")), frozenset(("CA", "CB1")),
    frozenset(("CA", "CB2")),
}


def requires_compiled_backend(cmd, system: str, state: int = 1) -> bool:
    model = cmd.get_model(system, int(state))
    return any(
        atom.resn.upper() in _PDB_SUGARS or atom.resn.upper() == "AIB"
        for atom in model.atom
    )


def _atom_key(atom):
    return (
        atom.model, atom.segi or "", atom.chain or "", atom.resi, atom.resn,
        atom.name, *(round(float(v), 3) for v in atom.coord),
    )


def _selection_key(atom):
    return (
        atom.segi or "", atom.chain or "", atom.resi, atom.resn, atom.name,
        *(round(float(v), 3) for v in atom.coord),
    )


def _surplus_anomeric_oxygens(model):
    """Find sugar O1 atoms retained despite an inter-residue C1 linkage."""
    residue_key = [
        (a.model, a.segi or "", a.chain or "", a.resi, a.resn) for a in model.atom
    ]
    remove = set()
    for i, atom in enumerate(model.atom):
        if atom.resn.upper() not in _PDB_SUGARS or atom.name != "C1":
            continue
        has_external_link = False
        same_residue_o1 = None
        for bond in model.bond:
            if i not in bond.index:
                continue
            neighbor = bond.index[1] if bond.index[0] == i else bond.index[0]
            if residue_key[neighbor] != residue_key[i]:
                has_external_link = True
            elif model.atom[neighbor].name == "O1":
                same_residue_o1 = neighbor
        if has_external_link and same_residue_o1 is not None:
            remove.add(same_residue_o1)
    return remove


def _invalid_aib_bonds(model):
    """Find impossible bonds PyMOL inferred from collapsed AIB coordinates."""
    invalid = []
    for bond in model.bond:
        left, right = (model.atom[i] for i in bond.index)
        if left.resn.upper() != "AIB" or right.resn.upper() != "AIB":
            continue
        same_residue = (
            left.model == right.model and left.segi == right.segi
            and left.chain == right.chain and left.resi == right.resi
        )
        if not same_residue:
            continue
        if (left.symbol or left.elem).upper() == "H" or (right.symbol or right.elem).upper() == "H":
            continue
        if frozenset((left.name, right.name)) not in _AIB_HEAVY_BONDS:
            invalid.append((left, right))
    return invalid


def _run(command, cwd: Path, stdin_path: Path | None = None) -> str:
    try:
        if stdin_path is None:
            completed = subprocess.run(
                command, cwd=cwd, text=True, capture_output=True, check=False
            )
        else:
            with stdin_path.open() as handle:
                completed = subprocess.run(
                    command, cwd=cwd, stdin=handle, text=True,
                    capture_output=True, check=False,
                )
    except FileNotFoundError as exc:
        raise MinimizationError(f"Required preparation program {command[0]!r} was not found.") from exc
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise MinimizationError(
            f"{command[0]} preparation failed (exit {completed.returncode}).\n" + output[-4000:]
        )
    return output


def _compile_topology(cmd, system: str, state: int, work: Path):
    temp_object = cmd.get_unused_name("_quick_relax_prepare")
    try:
        cmd.create(temp_object, system, state, 1)
        model = cmd.get_model(temp_object, 1)
        surplus = _surplus_anomeric_oxygens(model)
        if surplus:
            residues = sorted({
                f"{model.atom[i].resn} {model.atom[i].chain or '-'}:{model.atom[i].resi}"
                for i in surplus
            })
            preview = ", ".join(residues[:12])
            if len(residues) > 12:
                preview += f", and {len(residues) - 12} more"
            raise MinimizationError(
                "Invalid linked glycan: an anomeric O1 is still present even though C1 has an "
                f"inter-residue bond ({preview}). Clean the glycan connectivity first; no atoms "
                "were changed."
            )
        cleaned_model = cmd.get_model(temp_object, 1)
        cmd.save(str(work / "input.pdb"), temp_object, state=1)
    finally:
        cmd.delete(temp_object)

    cpptraj_input = work / "prepare.in"
    cpptraj_input.write_text(
        "parm input.pdb\n"
        "loadcrd input.pdb name Input\n"
        "prepareforleap crdset Input name Prepared leapunitname complex "
        "pdbout prepared.pdb out bonds.leap nowat noh determinesugarsby name "
        "problemout problems.pdb\n"
        "go\nquit\n"
    )
    cpptraj_log = _run(["cpptraj", "-i", "prepare.in"], work)
    fatal_lines = [line for line in cpptraj_log.splitlines() if line.startswith("Error:")]
    if fatal_lines:
        raise MinimizationError("Carbohydrate preparation reported:\n" + "\n".join(fatal_lines[:20]))

    prepared_residues = {a.resn.upper() for a in cleaned_model.atom}
    unsupported = []
    for match in re.finditer(r"Potential problem\s*:\s*(\S+).*unrecognized name", cpptraj_log):
        name = match.group(1).split("_")[0].upper()
        if name != "AIB":
            unsupported.append(name)
    if unsupported:
        raise MinimizationError(
            "Amber/GLYCAM preparation found unsupported residues: "
            + ", ".join(sorted(set(unsupported)))
        )

    has_aib = "AIB" in prepared_residues
    leap_lines = [
        "source leaprc.protein.ff14SB",
        "source leaprc.GLYCAM_06j-1",
    ]
    if has_aib:
        bridge = Path(__file__).with_name("data") / "frcmod.aib_ff14sb_bridge"
        leap_lines.extend([
            "loadamberparams frcmod.ff15ipq-m",
            "loadOff ff15ipq-m.lib",
            'set AIB.1.O type "O"',
            f'loadamberparams "{bridge}"',
        ])
    leap_lines.extend([
        "source bonds.leap",
        "check complex",
        "saveamberparm complex system.prmtop system.inpcrd",
        "savepdb complex system_prepared.pdb",
        "quit",
    ])
    (work / "build.leap").write_text("\n".join(leap_lines) + "\n")
    leap_log = _run(["tleap", "-f", "build.leap"], work)
    error_match = re.search(r"Exiting LEaP: Errors = (\d+)", leap_log)
    if error_match and int(error_match.group(1)):
        raise MinimizationError("LEaP topology compilation failed:\n" + leap_log[-5000:])
    if not (work / "system.prmtop").exists() or not (work / "system.inpcrd").exists():
        raise MinimizationError("LEaP did not produce a complete topology.")
    return cleaned_model, surplus, cpptraj_log, leap_log


def _coordinate_mapping(model, topology, positions):
    """Map PyMOL atoms to prepared particles by unchanged element/coordinates."""
    from openmm import unit
    from scipy.spatial import cKDTree

    prepared = np.asarray(positions.value_in_unit(unit.angstrom), dtype=float)
    atoms = list(topology.atoms())
    by_element = {}
    for i, atom in enumerate(atoms):
        by_element.setdefault(atom.element.symbol.upper(), []).append(i)
    trees = {
        element: (indices, cKDTree(prepared[indices]))
        for element, indices in by_element.items()
    }
    used = set()
    mapping = []
    for atom in model.atom:
        element = (atom.symbol or atom.elem).upper()
        if element not in trees:
            raise MinimizationError(f"Prepared topology lost element {element}.")
        indices, tree = trees[element]
        distances, local_indices = tree.query(np.asarray(atom.coord), k=min(8, len(indices)))
        distances = np.atleast_1d(distances)
        local_indices = np.atleast_1d(local_indices)
        chosen = None
        for distance, local_index in zip(distances, local_indices):
            candidate = indices[int(local_index)]
            if candidate not in used and float(distance) < 0.05:
                chosen = candidate
                break
        if chosen is None:
            raise MinimizationError(
                f"Could not safely map atom {atom.name} {atom.resn}{atom.resi} "
                "into the prepared topology."
            )
        used.add(chosen)
        mapping.append(chosen)
    return mapping


def _add_aib_geometry_guards(omm_system, topology):
    """Keep peptide bonds adjoining AIB planar during severe clash repair.

    Boltz-style inputs can place AIB methyl groups almost on top of each other.
    The force field then has a very large local gradient, and a positional
    backbone tether alone can let that strain escape by twisting peptide omega.
    This periodic guard has minima at both cis and trans, so it preserves peptide
    planarity without making a cis/trans assignment for the user.
    """
    from openmm import CustomAngleForce, CustomTorsionForce, unit

    atoms = list(topology.atoms())
    bonds = {atom.index: set() for atom in atoms}
    for bond in topology.bonds():
        bonds[bond[0].index].add(bond[1].index)
        bonds[bond[1].index].add(bond[0].index)

    residue_atoms = {
        residue: {atom.name: atom for atom in residue.atoms()}
        for chain in topology.chains() for residue in chain.residues()
    }
    guard = CustomTorsionForce("peptide_k*(1-cos(2*theta))")
    guard.addGlobalParameter(
        "peptide_k", 150.0 * unit.kilojoule_per_mole
    )
    added = set()
    aib_angle = CustomAngleForce("0.5*aib_angle_k*(theta-theta0)^2")
    aib_angle.addGlobalParameter(
        "aib_angle_k", 400.0 * unit.kilojoule_per_mole / unit.radian**2
    )
    aib_angle.addPerAngleParameter("theta0")
    angle_count = 0
    for residue, named in residue_atoms.items():
        if residue.name.upper() != "AIB" or not {"N", "CA", "C"} <= named.keys():
            continue
        if {"CB1", "CB2"} <= named.keys():
            aib_angle.addAngle(
                named["CB1"].index, named["CA"].index, named["CB2"].index,
                [109.5 * unit.degree],
            )
            angle_count += 1
        # Previous CA-C-N-CA peptide bond.
        for neighbor_index in bonds[named["N"].index]:
            neighbor = atoms[neighbor_index]
            previous = residue_atoms.get(neighbor.residue, {})
            if neighbor.name == "C" and "CA" in previous:
                indices = (
                    previous["CA"].index, neighbor.index,
                    named["N"].index, named["CA"].index,
                )
                if indices not in added:
                    guard.addTorsion(*indices)
                    added.add(indices)
        # Following CA-C-N-CA peptide bond.
        for neighbor_index in bonds[named["C"].index]:
            neighbor = atoms[neighbor_index]
            following = residue_atoms.get(neighbor.residue, {})
            if neighbor.name == "N" and "CA" in following:
                indices = (
                    named["CA"].index, named["C"].index,
                    neighbor.index, following["CA"].index,
                )
                if indices not in added:
                    guard.addTorsion(*indices)
                    added.add(indices)
    if added:
        omm_system.addForce(guard)
    if angle_count:
        omm_system.addForce(aib_angle)
    return len(added), angle_count


def _validate_aib_geometry(topology, coordinates):
    """Reject AIB output whose local covalent geometry remains implausible."""
    atoms = list(topology.atoms())
    xyz = np.asarray(coordinates, dtype=float)
    bonds = {atom.index: set() for atom in atoms}
    for bond in topology.bonds():
        bonds[bond[0].index].add(bond[1].index)
        bonds[bond[1].index].add(bond[0].index)
    residue_atoms = {
        residue: {atom.name: atom for atom in residue.atoms()}
        for chain in topology.chains() for residue in chain.residues()
    }

    def angle(i, j, k):
        left, right = xyz[i] - xyz[j], xyz[k] - xyz[j]
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def dihedral(i, j, k, l):
        b0 = -(xyz[j] - xyz[i])
        b1 = xyz[k] - xyz[j]
        b2 = xyz[l] - xyz[k]
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))

    issues = []
    for residue, named in residue_atoms.items():
        if residue.name.upper() != "AIB" or not {"N", "CA", "C"} <= named.keys():
            continue
        label = f"AIB {residue.chain.id or '-'}:{residue.id}"
        if {"CB1", "CB2"} <= named.keys():
            value = angle(named["CB1"].index, named["CA"].index, named["CB2"].index)
            if not 100.0 <= value <= 120.0:
                issues.append(f"{label} CB1-CA-CB2 angle is {value:.1f} degrees")
        peptide_torsions = []
        for neighbor_index in bonds[named["N"].index]:
            neighbor = atoms[neighbor_index]
            previous = residue_atoms.get(neighbor.residue, {})
            if neighbor.name == "C" and "CA" in previous:
                peptide_torsions.append(dihedral(
                    previous["CA"].index, neighbor.index,
                    named["N"].index, named["CA"].index,
                ))
        for neighbor_index in bonds[named["C"].index]:
            neighbor = atoms[neighbor_index]
            following = residue_atoms.get(neighbor.residue, {})
            if neighbor.name == "N" and "CA" in following:
                peptide_torsions.append(dihedral(
                    named["CA"].index, named["C"].index,
                    neighbor.index, following["CA"].index,
                ))
        for value in peptide_torsions:
            distance_from_planar = min(abs(value), abs(180.0 - abs(value)))
            if distance_from_planar > 20.0:
                issues.append(f"{label} peptide omega is {value:.1f} degrees")
    if issues:
        raise MinimizationError(
            "AIB geometry validation failed after relaxation: " + "; ".join(issues)
        )


def relax_compiled(
    selection="sele", system="all", state=1, mode="quick", quiet=0, _cmd=None,
    max_iterations=None, restraint=1000.0,
):
    from pymol import cmd as pymol_cmd
    from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, unit
    from openmm.app import AmberInpcrdFile, AmberPrmtopFile, NoCutoff, OBC2, PDBFile, Simulation

    cmd = _cmd or pymol_cmd
    state = int(state)
    mode = str(mode).lower()
    if mode not in {"quick", "thorough"}:
        raise MinimizationError("mode must be quick or thorough")
    selected_keys = {
        _selection_key(a) for a in cmd.get_model(f"({system}) and ({selection})", state).atom
    }
    if not selected_keys:
        raise MinimizationError("The movable selection contains no atoms.")

    with tempfile.TemporaryDirectory(prefix="pymol-quick-relax-") as temp:
        work = Path(temp)
        cleaned_model, surplus, _, _ = _compile_topology(cmd, system, state, work)
        prmtop = AmberPrmtopFile(str(work / "system.prmtop"))
        inpcrd = AmberInpcrdFile(str(work / "system.inpcrd"))
        mapping = _coordinate_mapping(cleaned_model, prmtop.topology, inpcrd.positions)
        selected_clean_positions = {
            i for i, atom in enumerate(cleaned_model.atom)
            if _selection_key(atom) in selected_keys
        }
        selected_particles = {mapping[i] for i in selected_clean_positions}
        selected_residues = {
            list(prmtop.topology.atoms())[i].residue for i in selected_particles
        }
        for atom in prmtop.topology.atoms():
            if atom.residue in selected_residues:
                selected_particles.add(atom.index)

        system_kwargs = {
            "nonbondedMethod": NoCutoff,
            "constraints": None,
            "removeCMMotion": False,
        }
        if mode == "thorough":
            system_kwargs["implicitSolvent"] = OBC2
        omm_system = prmtop.createSystem(**system_kwargs)
        _add_aib_geometry_guards(omm_system, prmtop.topology)
        reference = inpcrd.positions.value_in_unit(unit.nanometer)
        outside = CustomExternalForce("outside_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        outside.addGlobalParameter(
            "outside_k", float(restraint) * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for name in ("x0", "y0", "z0"):
            outside.addPerParticleParameter(name)
        for atom in prmtop.topology.atoms():
            if atom.index not in selected_particles:
                outside.addParticle(atom.index, [float(v) for v in reference[atom.index]])
        omm_system.addForce(outside)

        backbone = CustomExternalForce(
            "backbone_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        backbone.addGlobalParameter(
            "backbone_k", 500 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for name in ("x0", "y0", "z0"):
            backbone.addPerParticleParameter(name)
        for atom in prmtop.topology.atoms():
            if atom.residue.name.upper() in _STANDARD_PROTEIN and atom.name in {
                "N", "CA", "C", "O", "OXT"
            }:
                backbone.addParticle(atom.index, [float(v) for v in reference[atom.index]])
        omm_system.addForce(backbone)

        movable = None
        if mode == "thorough":
            movable = CustomExternalForce("movable_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            movable.addGlobalParameter(
                "movable_k", 5000 * unit.kilojoule_per_mole / unit.nanometer**2
            )
            for name in ("x0", "y0", "z0"):
                movable.addPerParticleParameter(name)
            atoms = list(prmtop.topology.atoms())
            for index in selected_particles:
                if atoms[index].element.symbol != "H":
                    movable.addParticle(index, [float(v) for v in reference[index]])
            omm_system.addForce(movable)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds
        )
        platform = Platform.getPlatformByName("CPU")
        simulation = Simulation(prmtop.topology, omm_system, integrator, platform)
        simulation.context.setPositions(inpcrd.positions)
        initial = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        iterations = int(max_iterations or (100 if mode == "quick" else 1000))
        if mode == "thorough":
            simulation.minimizeEnergy(tolerance=50, maxIterations=300)
            simulation.context.setParameter("movable_k", 100.0)
            simulation.minimizeEnergy(tolerance=20, maxIterations=500)
            simulation.context.setParameter("movable_k", 0.0)
            simulation.minimizeEnergy(tolerance=10, maxIterations=iterations)
        else:
            simulation.minimizeEnergy(tolerance=25, maxIterations=iterations)
        final_state = simulation.context.getState(getPositions=True, getEnergy=True)
        final = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = final_state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        _validate_aib_geometry(prmtop.topology, final_positions)
        output_coordinates = [final_positions[i] for i in mapping]

        cmd.load_coords(output_coordinates, system, state=state)
        # Collapsed AIB methyl groups can be close enough for PyMOL to infer
        # impossible N-CB or CB-CB bonds. Amber minimized the correct template;
        # remove those stale display/connectivity bonds only after validation.
        for left, right in _invalid_aib_bonds(cmd.get_model(system, state)):
            cmd.unbond(
                f"{left.model} and index {left.index}",
                f"{right.model} and index {right.index}",
            )
        cmd.rebuild()

    result = MinimizationResult(
        initial_energy_kj_mol=float(initial), final_energy_kj_mol=float(final),
        atom_count=prmtop.topology.getNumAtoms(),
        movable_atom_count=len(selected_particles), platform=platform.getName(),
    )
    if not _as_bool(quiet):
        print(
            f"{mode.title()} Relax: {result.movable_atom_count}/{result.atom_count} atoms movable; "
            "complete Amber/GLYCAM topology, "
            + ("OBC2 implicit solvent" if mode == "thorough" else "fast nonperiodic field")
        )
    return result
