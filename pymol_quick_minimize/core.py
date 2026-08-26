"""OpenMM-backed restrained minimization of a PyMOL selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from io import StringIO
import os
from typing import Any


class MinimizationError(RuntimeError):
    """A user-facing error raised when a structure cannot be minimized safely."""


@dataclass(frozen=True)
class MinimizationResult:
    initial_energy_kj_mol: float
    final_energy_kj_mol: float
    atom_count: int
    movable_atom_count: int
    platform: str


# Residues covered by the Amber protein/nucleic-acid/water files. Variants are
# intentionally included here only for classifying disconnected small molecules;
# OpenMM remains the authority on whether a residue actually has a template.
_BIOPOLYMER_RESIDUES = {
    "ALA", "ARG", "ASH", "ASN", "ASP", "CYS", "CYM", "CYX", "GLH", "GLN",
    "GLU", "GLY", "HID", "HIE", "HIP", "HIS", "ILE", "LEU", "LYN", "LYS",
    "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "A", "C", "G", "U", "DA", "DC", "DG", "DT", "HOH", "WAT",
}

_PROTEIN_BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _formal_charge(atom: Any) -> int:
    value = getattr(atom, "formal_charge", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _build_topology(model: Any):
    from openmm.app import Element, Topology

    topology = Topology()
    chains = {}
    residues = {}
    omm_atoms = []

    for position, atom in enumerate(model.atom):
        chain_key = (atom.model or "", atom.segi or "", atom.chain or "")
        chain = chains.get(chain_key)
        if chain is None:
            chain = topology.addChain(atom.chain or None)
            chains[chain_key] = chain

        residue_key = (chain_key, atom.resi, atom.resn)
        residue = residues.get(residue_key)
        if residue is None:
            residue = topology.addResidue(atom.resn or "UNK", chain, id=atom.resi or None)
            residues[residue_key] = residue

        symbol = (atom.symbol or atom.elem or "").strip().capitalize()
        try:
            element = Element.getBySymbol(symbol)
        except Exception as exc:
            raise MinimizationError(
                f"Atom {atom.name!r} in {atom.resn}{atom.resi} has unsupported element {symbol!r}."
            ) from exc
        # Numeric, globally unique ids survive OpenMM Modeller's topology copy.
        omm_atoms.append(topology.addAtom(atom.name or symbol, element, residue, id=str(position + 1)))

    bond_types = {1: None, 2: "Double", 3: "Triple", 4: "Aromatic"}
    from openmm.app import Double, Triple, Aromatic
    type_objects = {"Double": Double, "Triple": Triple, "Aromatic": Aromatic}
    for bond in model.bond:
        order = int(round(getattr(bond, "order", 1) or 1))
        topology.addBond(
            omm_atoms[bond.index[0]],
            omm_atoms[bond.index[1]],
            type=type_objects.get(bond_types.get(order)),
            order=1.5 if order == 4 else order,
        )

    return topology


def _small_molecules_from_model(model: Any):
    """Convert disconnected non-biopolymer components into OpenFF molecules."""
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    adjacency = [set() for _ in model.atom]
    for bond in model.bond:
        i, j = bond.index
        adjacency[i].add(j)
        adjacency[j].add(i)

    unseen = set(range(len(model.atom)))
    components = []
    while unseen:
        root = unseen.pop()
        component = {root}
        stack = [root]
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    molecules = []
    for component in components:
        residue_names = {model.atom[i].resn.upper() for i in component}
        if residue_names <= _BIOPOLYMER_RESIDUES:
            continue
        if residue_names & _BIOPOLYMER_RESIDUES:
            # A covalently attached ligand/PTM cannot safely be treated as a
            # disconnected OpenFF molecule. OpenMM will report the missing template.
            continue

        rw = Chem.RWMol()
        old_to_new = {}
        for old_index in component:
            source = model.atom[old_index]
            atom = Chem.Atom((source.symbol or source.elem).strip().capitalize())
            atom.SetFormalCharge(_formal_charge(source))
            atom.SetNoImplicit(True)
            atom.SetProp("_Name", source.name)
            old_to_new[old_index] = rw.AddAtom(atom)

        for bond in model.bond:
            i, j = bond.index
            if i not in old_to_new or j not in old_to_new:
                continue
            order = int(round(getattr(bond, "order", 1) or 1))
            bond_type = {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                4: Chem.BondType.AROMATIC,
            }.get(order, Chem.BondType.SINGLE)
            rw.AddBond(old_to_new[i], old_to_new[j], bond_type)

        mol = rw.GetMol()
        conformer = Chem.Conformer(len(component))
        for old_index, new_index in old_to_new.items():
            x, y, z = model.atom[old_index].coord
            conformer.SetAtomPosition(new_index, Point3D(float(x), float(y), float(z)))
        mol.AddConformer(conformer)
        try:
            if any(
                int(round(getattr(bond, "order", 1) or 1)) == 4
                and (len(adjacency[bond.index[0]]) == 1 or len(adjacency[bond.index[1]]) == 1)
                for bond in model.bond
                if bond.index[0] in old_to_new and bond.index[1] in old_to_new
            ):
                raise ValueError("terminal delocalized bond requires bond-order inference")
            Chem.SanitizeMol(mol)
        except Exception:
            # PyMOL uses bond order 4 both for aromatic bonds and for some
            # delocalized/resonance bonds (notably terminal carboxylates). Rebuild
            # connectivity and let RDKit infer bond orders from valence and 3D
            # coordinates when the literal aromatic interpretation is invalid.
            from rdkit.Chem import rdDetermineBonds

            fallback = Chem.RWMol()
            fallback_old_to_new = {}
            for old_index in component:
                source = model.atom[old_index]
                atom = Chem.Atom((source.symbol or source.elem).strip().capitalize())
                atom.SetFormalCharge(_formal_charge(source))
                atom.SetNoImplicit(True)
                atom.SetProp("_Name", source.name)
                fallback_old_to_new[old_index] = fallback.AddAtom(atom)
            for bond in model.bond:
                i, j = bond.index
                if i in fallback_old_to_new and j in fallback_old_to_new:
                    fallback.AddBond(
                        fallback_old_to_new[i], fallback_old_to_new[j], Chem.BondType.SINGLE
                    )
            fallback_mol = fallback.GetMol()
            fallback_conformer = Chem.Conformer(len(component))
            for old_index, new_index in fallback_old_to_new.items():
                x, y, z = model.atom[old_index].coord
                fallback_conformer.SetAtomPosition(
                    new_index, Point3D(float(x), float(y), float(z))
                )
            fallback_mol.AddConformer(fallback_conformer)
            try:
                stated_charge = sum(_formal_charge(model.atom[i]) for i in component)
                charge_candidates = [stated_charge]
                if stated_charge == 0:
                    # PDB/mmCIF exports frequently omit ligand formal charges.
                    # Try common small net charges, preferring anions before
                    # cations because deprotonated oxygens are common.
                    charge_candidates.extend([-1, 1, -2, 2, -3, 3])
                last_error = None
                for total_charge in charge_candidates:
                    candidate = Chem.Mol(fallback_mol)
                    try:
                        rdDetermineBonds.DetermineBondOrders(
                            candidate, charge=total_charge, allowChargedFragments=True
                        )
                        Chem.SanitizeMol(candidate)
                        fallback_mol = candidate
                        break
                    except Exception as charge_error:
                        last_error = charge_error
                else:
                    raise last_error
                mol = fallback_mol
            except Exception as exc:
                names = ", ".join(sorted(residue_names))
                raise MinimizationError(
                    f"Could not determine chemically valid bond orders/charges for {names}. "
                    "Load that ligand from SDF or MOL2 (not PDB), then try again."
                ) from exc
        try:
            molecules.append(
                Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
            )
        except Exception as exc:
            names = ", ".join(sorted(residue_names))
            raise MinimizationError(
                f"Could not determine chemically valid bond orders/charges for {names}. "
                "Load that ligand from SDF or MOL2 (not PDB), then try again."
            ) from exc

    return molecules


def _energy(context) -> float:
    from openmm import unit

    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _repair_missing_heavy_atoms(topology, positions):
    """Use PDBFixer to add missing atoms without guessing missing residues."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    buffer = StringIO()
    PDBFile.writeFile(topology, positions, buffer, keepIds=True)
    buffer.seek(0)
    fixer = PDBFixer(pdbfile=buffer)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    if fixer.missingAtoms or fixer.missingTerminals:
        fixer.addMissingAtoms()
    return fixer.topology, fixer.positions


def _template_cache_path(cache: str):
    if cache and cache != "auto":
        path = Path(cache).expanduser()
    elif cache == "auto":
        if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
            root = Path(os.environ["LOCALAPPDATA"])
        elif os.environ.get("XDG_CACHE_HOME"):
            root = Path(os.environ["XDG_CACHE_HOME"])
        else:
            root = Path.home() / ".cache"
        path = root / "pymol-quick-minimize" / "openff-templates.json"
    else:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _topology_atom_key(atom) -> tuple[int, int, str]:
    """Identity stable across OpenMM Modeller topology reconstruction."""
    return atom.residue.chain.index, atom.residue.index, atom.name


def minimize_selection(
    selection: str = "sele",
    system: str = "all",
    state: int = 1,
    max_iterations: int = 500,
    tolerance: float = 10.0,
    restraint: float = 1000.0,
    add_hydrogens: Any = 1,
    forcefield: str = "openff-2.2.1",
    cache: str = "auto",
    implicit_solvent: Any = 1,
    relax_mode: str = "quick",
    quiet: Any = 0,
    _cmd=None,
) -> MinimizationResult:
    """Minimize ``selection`` while restraining the rest of ``system``.

    This function is also registered as the PyMOL command ``quick_minimize``.
    Energies are in kJ/mol, tolerance in kJ/mol/nm, and restraint strength in
    kJ/mol/nm^2.
    """
    from pymol import cmd as pymol_cmd
    from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, Vec3, unit
    from openmm.app import Modeller, NoCutoff, Simulation
    from openmmforcefields.generators import SystemGenerator

    cmd = _cmd or pymol_cmd
    state = int(state)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    restraint = float(restraint)
    quiet = _as_bool(quiet)
    implicit_solvent = _as_bool(implicit_solvent)
    relax_mode = str(relax_mode).strip().lower()
    if relax_mode not in {"quick", "thorough"}:
        raise MinimizationError("relax_mode must be 'quick' or 'thorough'.")

    if cmd.count_atoms(selection) == 0:
        raise MinimizationError(f"Selection {selection!r} contains no atoms.")
    if cmd.count_atoms(system) == 0:
        raise MinimizationError(f"System selection {system!r} contains no atoms.")

    if _as_bool(add_hydrogens):
        # PyMOL generally does a useful job on standalone organic molecules, and
        # those explicit H atoms are required to preserve the ligand's identity.
        # Protein H atoms are added below by OpenMM, whose residue templates know
        # the correct terminal/protonation naming and bonding patterns.
        cmd.h_add(f"({system}) and not polymer", state=state)

    model = cmd.get_model(system, state)
    if not model.atom:
        raise MinimizationError("No atoms were available after structure preparation.")

    system_indices = [(atom.model, atom.index) for atom in model.atom]
    movable_indices = set(cmd.index(f"({system}) and ({selection})"))
    movable_positions = {i for i, identity in enumerate(system_indices) if identity in movable_indices}
    if not movable_positions:
        raise MinimizationError("The movable selection does not overlap the system selection.")

    topology = _build_topology(model)
    positions = [Vec3(*map(float, atom.coord)) for atom in model.atom] * unit.angstrom
    original_atom_keys = [_topology_atom_key(atom) for atom in topology.atoms()]
    if len(set(original_atom_keys)) != len(original_atom_keys):
        raise MinimizationError(
            "The system contains duplicate atom names within a residue. Rename atoms uniquely "
            "before minimization so coordinates can be mapped back safely."
        )
    movable_original_keys = {
        original_atom_keys[i] for i in movable_positions
    }
    movable_residue_keys = {
        key[:2] for key in movable_original_keys
    }
    molecules = _small_molecules_from_model(model)
    try:
        topology, positions = _repair_missing_heavy_atoms(topology, positions)
    except Exception as exc:
        raise MinimizationError("Automatic heavy-atom/terminal repair failed: " + str(exc)) from exc

    cache_path = _template_cache_path(cache)
    forcefield_files = ["amber/protein.ff14SB.xml", "amber/tip3p_standard.xml"]
    if implicit_solvent:
        forcefield_files.append("implicit/obc2.xml")
    generator = SystemGenerator(
        forcefields=forcefield_files,
        small_molecule_forcefield=forcefield,
        forcefield_kwargs={"constraints": None, "rigidWater": False, "removeCMMotion": False},
        nonperiodic_forcefield_kwargs={"nonbondedMethod": NoCutoff},
        molecules=molecules,
        cache=cache_path,
    )
    if _as_bool(add_hydrogens):
        modeller = Modeller(topology, positions)
        try:
            modeller.addHydrogens(generator.forcefield)
        except Exception as exc:
            raise MinimizationError(
                "Hydrogen placement failed. Check residue names, missing heavy atoms, and ligand "
                "bond orders. Original error: " + str(exc)
            ) from exc
        topology, positions = modeller.topology, modeller.positions

    # Modeller inserts and renumbers protein hydrogens. Chain/residue indices and
    # unique atom names are stable; generated H atoms inherit mobility by residue.
    position_by_key = {
        _topology_atom_key(atom): i for i, atom in enumerate(topology.atoms())
    }
    missing_keys = [key for key in original_atom_keys if key not in position_by_key]
    if missing_keys:
        raise MinimizationError(
            "OpenMM structure preparation changed original atom identities; refusing unsafe "
            "coordinate write-back."
        )
    movable_positions = {
        i for i, atom in enumerate(topology.atoms())
        if _topology_atom_key(atom) in movable_original_keys
        or _topology_atom_key(atom)[:2] in movable_residue_keys
    }
    try:
        omm_system = generator.create_system(topology, molecules=molecules)
    except Exception as exc:
        raise MinimizationError(
            "Force-field assignment failed. Check protonation, explicit hydrogens, bond orders, "
            "metals, and covalently attached non-standard residues. Original error: " + str(exc)
        ) from exc

    if restraint > 0:
        restraint_force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint_force.addGlobalParameter("k", restraint * unit.kilojoule_per_mole / unit.nanometer**2)
        for parameter in ("x0", "y0", "z0"):
            restraint_force.addPerParticleParameter(parameter)
        positions_nm = positions.value_in_unit(unit.nanometer)
        for i, xyz in enumerate(positions_nm):
            if i not in movable_positions:
                restraint_force.addParticle(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
        omm_system.addForce(restraint_force)

    movable_tether = None
    if relax_mode == "thorough":
        movable_tether = CustomExternalForce(
            "movable_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        movable_tether.addGlobalParameter(
            "movable_k", 5000 * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for parameter in ("x0", "y0", "z0"):
            movable_tether.addPerParticleParameter(parameter)
        positions_nm = positions.value_in_unit(unit.nanometer)
        topology_atoms = list(topology.atoms())
        for i in movable_positions:
            if topology_atoms[i].element.symbol != "H":
                xyz = positions_nm[i]
                movable_tether.addParticle(i, [float(xyz[0]), float(xyz[1]), float(xyz[2])])
        omm_system.addForce(movable_tether)

    # Keep the experimental/model backbone close to its input even when a
    # residue-wise pocket selection includes backbone atoms (or selection=all).
    backbone_tether = CustomExternalForce(
        "backbone_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    backbone_tether.addGlobalParameter(
        "backbone_k", 500 * unit.kilojoule_per_mole / unit.nanometer**2
    )
    for parameter in ("x0", "y0", "z0"):
        backbone_tether.addPerParticleParameter(parameter)
    positions_nm = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.residue.name.upper() in _BIOPOLYMER_RESIDUES and atom.name in _PROTEIN_BACKBONE_NAMES:
            xyz = positions_nm[atom.index]
            backbone_tether.addParticle(
                atom.index, [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            )
    omm_system.addForce(backbone_tether)

    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName("CPU")
    simulation = Simulation(topology, omm_system, integrator, platform)
    simulation.context.setPositions(positions)
    initial_energy = _energy(simulation.context)
    if relax_mode == "thorough":
        simulation.minimizeEnergy(tolerance=50, maxIterations=min(300, max_iterations))
        simulation.context.setParameter("movable_k", 100.0)
        simulation.minimizeEnergy(tolerance=20, maxIterations=min(500, max_iterations))
        simulation.context.setParameter("movable_k", 0.0)
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)
    else:
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)
    final_energy = _energy(simulation.context)
    minimized = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    coordinates = minimized.value_in_unit(unit.angstrom)
    original_coordinates = [
        coordinates[position_by_key[key]] for key in original_atom_keys
    ]
    cmd.load_coords(original_coordinates, system, state=state)
    cmd.rebuild()

    result = MinimizationResult(
        initial_energy_kj_mol=initial_energy,
        final_energy_kj_mol=final_energy,
        atom_count=omm_system.getNumParticles(),
        movable_atom_count=len(movable_positions),
        platform=platform.getName(),
    )
    if not quiet:
        print(
            f"{relax_mode.title()} Relax: "
            f"{result.movable_atom_count}/{result.atom_count} atoms movable; "
            + ("OBC2 implicit solvent" if implicit_solvent else "fast nonperiodic field")
            + f" ({result.platform})"
        )
    return result


def quick_relax(selection="sele", system="all", quiet=0, _cmd=None, **kwargs):
    """Fast selected-atom relaxation in the field of the complete system."""
    from pymol import cmd as pymol_cmd
    from .amber_backend import relax_compiled, requires_compiled_backend

    # PyMOL's cmd.extend may inject its command instance as ``_self``. It is
    # control metadata, not a minimization option, so consume it at this public
    # wrapper boundary before dispatching to either chemistry backend.
    pymol_self = kwargs.pop("_self", None)
    active_cmd = _cmd or pymol_self or pymol_cmd
    # MOE-like behavior: with no active PyMOL selection, relax the complete
    # system instead of failing on the conventional but absent ``sele`` name.
    if str(selection).strip() == "sele" and "sele" not in active_cmd.get_names("selections"):
        selection = system
    if requires_compiled_backend(active_cmd, system, int(kwargs.get("state", 1))):
        return relax_compiled(
            selection, system, mode="quick", quiet=quiet, _cmd=active_cmd,
            state=kwargs.get("state", 1), max_iterations=kwargs.get("max_iterations"),
            restraint=kwargs.get("restraint", 1000.0),
        )
    kwargs.setdefault("max_iterations", 100)
    kwargs.setdefault("tolerance", 25.0)
    kwargs.setdefault("restraint", 1000.0)
    kwargs.setdefault("implicit_solvent", 0)
    kwargs["relax_mode"] = "quick"
    return minimize_selection(selection, system, quiet=quiet, _cmd=active_cmd, **kwargs)


def thorough_relax(selection="sele", system="all", quiet=0, _cmd=None, **kwargs):
    """Staged implicit-solvent relaxation for difficult clashes."""
    from pymol import cmd as pymol_cmd
    from .amber_backend import relax_compiled, requires_compiled_backend

    pymol_self = kwargs.pop("_self", None)
    active_cmd = _cmd or pymol_self or pymol_cmd
    if str(selection).strip() == "sele" and "sele" not in active_cmd.get_names("selections"):
        selection = system
    if requires_compiled_backend(active_cmd, system, int(kwargs.get("state", 1))):
        return relax_compiled(
            selection, system, mode="thorough", quiet=quiet, _cmd=active_cmd,
            state=kwargs.get("state", 1), max_iterations=kwargs.get("max_iterations"),
            restraint=kwargs.get("restraint", 1000.0),
        )
    kwargs.setdefault("max_iterations", 1000)
    kwargs.setdefault("tolerance", 10.0)
    kwargs.setdefault("restraint", 1000.0)
    kwargs.setdefault("implicit_solvent", 1)
    kwargs["relax_mode"] = "thorough"
    return minimize_selection(selection, system, quiet=quiet, _cmd=active_cmd, **kwargs)
