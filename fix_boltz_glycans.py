#!/usr/bin/env python
"""Local, conservative repair of the glycan artifacts seen in these Boltz models.

Run in the pymol-quick-minimize environment. No network services are used.

    python dev/fix_boltz_glycans.py model.cif --dry-run
    python dev/fix_boltz_glycans.py model.cif
    python dev/fix_boltz_glycans.py merged.cif --reference correctly_numbered.cif

The default output is <stem>_glycan_fixed.<suffix>. Inputs and existing outputs
are never overwritten. Retained atom coordinates are copied without alteration.

O1 is removed only from a verified pyranose ring whose C1 already has a direct
glycosidic bond to another sugar's O2/O3/O4/O6 or an asparagine ND2. Free terminal
O1 and O1-mediated linkages are retained. Ambiguous bonds cause an error.
Direct, covalent-length contacts are assumed to be the intended linkages; this
does not establish the biological correctness of a predicted glycan topology.

Merged residues need a reference with correct sugar identities and numbering.
Atom names/order, element inventories, linkages, and ring handedness must match that
reference (allowing surplus O1/HO1). Atom inventories alone cannot distinguish
mannose from glucose, so residue types are never guessed from those inventories.
This is an input-repair utility, not a general carbohydrate or protonation tool.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from io import StringIO
import json
import math
from pathlib import Path
import sys


SUGARS = {"NAG", "NDG", "MAN", "BMA", "GLC", "BGC", "GAL", "FUC"}
RING = ("C1", "C2", "C3", "C4", "C5", "O5")


class RepairError(ValueError):
    pass


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    element: str
    chain: str
    resi: str
    resn: str
    insertion: str
    alt: str
    xyz: tuple[float, float, float]
    row: int

    @property
    def residue(self):
        return self.chain, self.resi, self.insertion, self.resn

    @property
    def label(self):
        return f"{self.resn} {self.chain or '-'}:{self.resi}{self.insertion}/{self.name}"


class Structure:
    def __init__(self, path: Path):
        self.path = path
        self.atoms = []
        self.blocks = None
        self.lines = None
        if path.suffix.lower() in {".cif", ".mmcif"}:
            self._read_cif()
        elif path.suffix.lower() == ".pdb":
            self._read_pdb()
        else:
            raise RepairError("Input must be .cif, .mmcif, or .pdb")
        if not self.atoms:
            raise RepairError("Input contains no atoms")
        if len({a.serial for a in self.atoms}) != len(self.atoms):
            raise RepairError("Atom serial IDs must be unique")
        if any(not all(math.isfinite(x) for x in a.xyz) for a in self.atoms):
            raise RepairError("Non-finite input coordinates")
        if any(a.alt for a in self.atoms if a.resn in SUGARS):
            raise RepairError("Resolve glycan alternate conformations before repair")

    def _read_cif(self):
        from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

        self.blocks = []
        with self.path.open() as handle:
            PdbxReader(handle).read(self.blocks)
        if len(self.blocks) != 1:
            raise RepairError("Expected exactly one mmCIF data block")
        site = self.blocks[0].getObj("atom_site")
        if site is None:
            raise RepairError("mmCIF has no atom_site table")
        columns = site.getAttributeList()
        models = set()
        for index, row in enumerate(site.getRowList()):
            def get(*names, default=""):
                for name in names:
                    if name in columns and row[columns.index(name)] not in {".", "?", ""}:
                        return row[columns.index(name)]
                return default

            models.add(get("pdbx_PDB_model_num", default="1"))
            self.atoms.append(Atom(
                int(get("id")), get("auth_atom_id", "label_atom_id"),
                get("type_symbol").upper(), get("auth_asym_id", "label_asym_id"),
                get("auth_seq_id", "label_seq_id"), get("auth_comp_id", "label_comp_id"),
                get("pdbx_PDB_ins_code"), get("label_alt_id"),
                tuple(float(get(name)) for name in ("Cartn_x", "Cartn_y", "Cartn_z")), index,
            ))
        if len(models) != 1:
            raise RepairError("Split multi-model mmCIF files before repair")

    def _read_pdb(self):
        self.lines = self.path.read_text().splitlines(keepends=True)
        if sum(line.startswith("MODEL ") for line in self.lines) > 1:
            raise RepairError("Split multi-model PDB files before repair")
        for index, line in enumerate(self.lines):
            if line[:6] not in {"ATOM  ", "HETATM"}:
                continue
            # The element field avoids confusing protein CA with calcium.
            if not line[76:78].strip():
                raise RepairError("PDB atoms require explicit element fields; use mmCIF instead")
            self.atoms.append(Atom(
                int(line[6:11]), line[12:16].strip(), line[76:78].strip().upper(),
                line[21:22].strip(), line[22:26].strip(), line[17:20].strip(),
                line[26:27].strip(), line[16:17].strip(),
                tuple(float(line[start:start+8]) for start in (30, 38, 46)), index,
            ))

    def render(self, atoms, deleted):
        original = {a.serial: a for a in self.atoms}
        modified = any(a.residue != original[a.serial].residue for a in atoms)
        if self.blocks is not None:
            from openmm.app.internal.pdbx.writer.PdbxWriter import PdbxWriter

            block = self.blocks[0]
            # Updating explicit links and branch annotations needs more than
            # atom_site edits. Reject instead of leaving inconsistent metadata.
            categories = ["struct_conn"]
            if modified:
                categories += ["pdbx_branch_scheme", "pdbx_nonpoly_scheme"]
            if deleted or modified:
                for name in categories:
                    obj = block.getObj(name)
                    if obj is not None and obj.getRowCount():
                        raise RepairError(f"Explicit {name} annotations require manual review")
            site = block.getObj("atom_site")
            for atom in atoms:
                if atom.residue == original[atom.serial].residue:
                    continue
                for name in ("auth_comp_id", "label_comp_id"):
                    site.setValue(atom.resn, name, atom.row)
                for name in ("auth_seq_id", "label_seq_id"):
                    site.setValue(atom.resi, name, atom.row)
                if site.hasAttribute("pdbx_PDB_ins_code"):
                    site.setValue(atom.insertion or "?", "pdbx_PDB_ins_code", atom.row)
            site.setRowList([row for index, row in enumerate(site.getRowList())
                             if self.atoms[index].serial not in deleted])
            buffer = StringIO()
            PdbxWriter(buffer).write(self.blocks)
            return buffer.getvalue()

        if (deleted or modified) and any(line.startswith("LINK  ") for line in self.lines):
            raise RepairError("Explicit PDB LINK annotations require manual review")
        by_serial = {a.serial: a for a in atoms}
        output = []
        for line in self.lines:
            record = line[:6]
            if record in {"ATOM  ", "HETATM", "ANISOU"}:
                serial = int(line[6:11])
                if serial in deleted:
                    continue
                atom = by_serial[serial]
                if len(atom.resi) > 4:
                    raise RepairError("Repaired residue number exceeds PDB field width")
                line = line[:17] + f"{atom.resn:>3}" + line[20:22] + f"{atom.resi:>4}" + (atom.insertion or " ") + line[27:]
            elif record == "CONECT":
                serials = [int(line[start:start+5]) for start in range(6, len(line.rstrip()), 5)
                           if line[start:start+5].strip()]
                if not serials or serials[0] in deleted:
                    continue
                serials = [serial for serial in serials if serial not in deleted]
                if len(serials) < 2:
                    continue
                line = "CONECT" + "".join(f"{serial:5d}" for serial in serials) + "\n"
            elif record == "MASTER" and (deleted or modified):
                continue  # Its old atom/connection counts no longer apply.
            output.append(line)
        return "".join(output)


def sugar_groups(atoms):
    groups = defaultdict(list)
    for atom in atoms:
        if atom.resn in SUGARS:
            groups[atom.residue].append(atom)
    return groups


def ring_handedness(group):
    """Check C2-C5 stereocenters before transferring reference residue names."""
    import numpy as np

    named = {a.name: a for a in group}
    result = []
    for center, neighbors in (
        ("C2", ("C1", "C3", "N2" if "N2" in named else "O2")),
        ("C3", ("C2", "C4", "O3")), ("C4", ("C3", "C5", "O4")),
        ("C5", ("C4", "O5", "C6")),
    ):
        if not {center, *neighbors} <= named.keys():
            raise RepairError("Reference reconstruction requires a complete supported sugar ring")
        origin = np.asarray(named[center].xyz)
        volume = float(np.linalg.det([np.asarray(named[n].xyz) - origin for n in neighbors]))
        if abs(volume) < 0.1:
            raise RepairError("Collapsed sugar stereocenter; cannot transfer reference identities")
        result.append(volume > 0)
    return tuple(result)


def restore_residue_ids(atoms, reference):
    groups = sugar_groups(atoms)
    bad = {key for key, group in groups.items()
           if len({a.name for a in group}) != len(group)}
    if not bad:
        return list(atoms), []
    if reference is None:
        raise RepairError("Merged/duplicate sugar atom names: supply --reference with correct sugar residue identities")
    reference_groups = sugar_groups(reference.atoms)
    if any(len({a.name for a in group}) != len(group) for group in reference_groups.values()):
        raise RepairError("The reference also contains merged/duplicate sugar atom names")
    result = {a.serial: a for a in atoms}
    changes = []
    affected_chains = {key[0] for key in bad}
    for chain in affected_chains:
        source = [a for a in atoms if a.chain == chain and a.resn in SUGARS]
        templates = [group for key, group in reference_groups.items() if key[0] == chain]
        # Only heavy atoms are matched. Explicit H in merged residues would
        # require additional protonation/name mapping, so fail conservatively.
        if any(a.element in {"H", "D"} for a in source):
            raise RepairError("Merged glycans with explicit hydrogens require manual review")
        source = [a for a in source if a.name != "O1"]
        templates = [[a for a in group if a.name != "O1" and a.element not in {"H", "D"}]
                     for group in templates]
        expected = [a for group in templates for a in group]
        signature = lambda items: [(a.name, a.element) for a in items]
        if not templates or signature(source) != signature(expected):
            raise RepairError(f"Sugar atom order/inventory in chain {chain!r} does not match the reference")
        offset = 0
        for template in templates:
            chunk = source[offset:offset+len(template)]
            offset += len(template)
            if ring_handedness(chunk) != ring_handedness(template):
                raise RepairError("Sugar ring handedness differs from reference; refusing residue reassignment")
            target = template[0]
            for atom in chunk:
                result[atom.serial] = replace(atom, resi=target.resi, insertion=target.insertion, resn=target.resn)
            # If present, O1 must occur inside its sugar's original contiguous
            # record block. Never assign it by nearest coordinates.
            first_row = chunk[0].row
            next_row = source[offset].row if offset < len(source) else max(a.row for a in atoms if a.chain == chain and a.resn in SUGARS) + 1
            extras = [a for a in atoms if a.chain == chain and a.resn in SUGARS
                      and a.name == "O1" and first_row <= a.row < next_row]
            if len(extras) > 1:
                raise RepairError("Ambiguous O1 assignment in merged residue")
            for atom in extras:
                result[atom.serial] = replace(atom, resi=target.resi, insertion=target.insertion, resn=target.resn)
            changes.append({"chain": chain, "resi": target.resi, "resn": target.resn,
                            "atom_count": len(chunk) + len(extras)})
    repaired = [result[a.serial] for a in atoms]
    if any(len({a.name for a in group}) != len(group) for group in sugar_groups(repaired).values()):
        raise RepairError("Reference reconstruction left duplicate atom names")
    return repaired, changes


def read_bonds(path, atoms):
    import pymol2

    by_serial = {a.serial: a for a in atoms}
    adjacency = {a.serial: set() for a in atoms}
    with pymol2.PyMOL() as session:
        cmd = session.cmd
        cmd.load(str(path.resolve()), "repair_input")
        if cmd.count_states("repair_input") != 1:
            raise RepairError("Expected exactly one model")
        model = cmd.get_model("repair_input")
        if len(model.atom) != len(atoms) or {a.id for a in model.atom} != set(by_serial):
            raise RepairError("PyMOL changed atom identities on import")
        for atom in model.atom:
            original = by_serial[atom.id]
            if atom.name != original.name or math.dist(atom.coord, original.xyz) > 0.002:
                raise RepairError("PyMOL atom identity/coordinate mismatch")
        for bond in model.bond:
            left, right = (model.atom[i].id for i in bond.index)
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def find_surplus_o1(atoms, bonds):
    by_serial = {a.serial: a for a in atoms}
    deleted = set()
    changes = []
    for group in sugar_groups(atoms).values():
        named = {a.name: a for a in group}
        if len(named) != len(group):
            raise RepairError("Duplicate sugar atom names remain")
        if "O1" not in named:
            continue
        if not set(RING) <= named.keys():
            raise RepairError(f"Incomplete ring at {group[0].label}")
        for left, right in zip(RING, RING[1:] + RING[:1]):
            if named[right].serial not in bonds[named[left].serial]:
                raise RepairError(f"Broken sugar ring at {group[0].label}")
        c1, o1 = named["C1"], named["O1"]
        heavy = {i for i in bonds[c1.serial] if by_serial[i].element not in {"H", "D"}}
        outside = [by_serial[i] for i in heavy if by_serial[i].residue != c1.residue]
        if not outside:
            continue  # A free anomeric OH or an O1-mediated linkage is retained.
        if len(outside) != 1:
            raise RepairError(f"Ambiguous external C1 bonds at {c1.label}")
        acceptor = outside[0]
        known = (acceptor.resn in SUGARS and acceptor.name in {"O2", "O3", "O4", "O6"}
                 and acceptor.element == "O") or (
                     acceptor.resn in {"ASN", "NLN"} and acceptor.name == "ND2" and acceptor.element == "N")
        expected = {named["C2"].serial, named["O5"].serial, o1.serial, acceptor.serial}
        if not known or heavy != expected or c1.element != "C" or o1.element != "O":
            raise RepairError(f"Unrecognized/ambiguous glycosidic geometry at {c1.label}")
        if not 1.1 <= math.dist(c1.xyz, acceptor.xyz) <= 1.8 or not 1.1 <= math.dist(c1.xyz, o1.xyz) <= 1.8:
            raise RepairError(f"Implausible glycosidic bond length at {c1.label}")
        o1_heavy = {i for i in bonds[o1.serial] if by_serial[i].element not in {"H", "D"}}
        if c1.serial not in o1_heavy or not o1_heavy <= {c1.serial, acceptor.serial}:
            raise RepairError(f"O1 has another covalent attachment at {o1.label}; refusing deletion")
        hydrogens = {i for i in bonds[o1.serial] if by_serial[i].element in {"H", "D"}}
        if any(bonds[i] != {o1.serial} for i in hydrogens):
            raise RepairError(f"Ambiguous O1 hydrogen connectivity at {o1.label}")
        deleted.update({o1.serial, *hydrogens})
        changes.append({"remove": o1.label, "serial": o1.serial,
                        "link_to": acceptor.label, "hydrogens_removed": len(hydrogens),
                        "c1_acceptor_distance_angstrom": math.dist(c1.xyz, acceptor.xyz),
                        "c1_o1_distance_angstrom": math.dist(c1.xyz, o1.xyz),
                        "acceptor_serial": acceptor.serial})
    # Check the completed repair, since another deleted O1 can also contact an
    # acceptor. Do not silently keep an overbonded glycosidic O or N.
    for change in changes:
        acceptor = change["acceptor_serial"]
        if len(bonds[acceptor] - deleted) != 2:
            raise RepairError(f"Acceptor remains over/underbonded at {by_serial[acceptor].label}")
    return deleted, changes


def check_reference_linkages(atoms, bonds, reference):
    """Also check linkage topology and the C1 anomer before copying labels."""
    import numpy as np

    reference_bonds = read_bonds(reference.path, reference.atoms)

    def fingerprint(items, adjacency):
        by_id = {a.serial: a for a in items}
        links = set()
        anomers = {}
        for atom in items:
            if atom.resn not in SUGARS or atom.name == "O1" or atom.element in {"H", "D"}:
                continue
            for neighbor_id in adjacency[atom.serial]:
                neighbor = by_id[neighbor_id]
                if neighbor.residue == atom.residue or neighbor.name == "O1" or neighbor.element in {"H", "D"}:
                    continue
                links.add(tuple(sorted(((*atom.residue, atom.name), (*neighbor.residue, neighbor.name)))))
        for key, group in sugar_groups(items).items():
            named = {a.name: a for a in group}
            c1 = named["C1"]
            external = [by_id[i] for i in adjacency[c1.serial]
                        if by_id[i].residue != c1.residue and by_id[i].element not in {"H", "D"}]
            if len(external) == 1:
                third = external[0]
            elif not external and "O1" in named:
                third = named["O1"]
            else:
                raise RepairError(f"Cannot establish C1 anomer at {c1.label}")
            origin = np.asarray(c1.xyz)
            volume = float(np.linalg.det([np.asarray(a.xyz) - origin
                                          for a in (named["C2"], named["O5"], third)]))
            if abs(volume) < 0.1:
                raise RepairError(f"Collapsed anomeric center at {c1.label}")
            anomers[key] = volume > 0
        return links, anomers

    if fingerprint(atoms, bonds) != fingerprint(reference.atoms, reference_bonds):
        raise RepairError("Sugar linkages or C1 anomers differ from the reference; refusing residue reassignment")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--reference", type=Path, help="Correct sugar residue identities for merged glycans; coordinates are not copied")
    parser.add_argument("--dry-run", action="store_true", help="Print the proposed changes without writing any files")
    parser.add_argument("--report", type=Path, help="Optional local JSON audit report")
    args = parser.parse_args(argv)
    output = args.output or args.input.with_name(args.input.stem + "_glycan_fixed" + args.input.suffix)
    try:
        if output.suffix.lower() != args.input.suffix.lower():
            raise RepairError("Output must use the same format/extension as the input")
        protected = {p.resolve() for p in (args.input, args.reference) if p is not None}
        destinations = [output] + ([args.report] if args.report else [])
        if len({p.resolve() for p in destinations}) != len(destinations):
            raise RepairError("Output and report paths must differ")
        for path in destinations:
            if path.resolve() in protected or path.exists():
                raise RepairError(f"Refusing to overwrite an input, reference, or existing output: {path}")
            if not path.parent.is_dir():
                raise RepairError(f"Output directory does not exist: {path.parent}")
        structure = Structure(args.input)
        reference = Structure(args.reference) if args.reference else None
        atoms, renamed = restore_residue_ids(structure.atoms, reference)
        bonds = read_bonds(args.input, atoms)
        if renamed:
            check_reference_linkages(atoms, bonds, reference)
        deleted, removals = find_surplus_o1(atoms, bonds)
        # Render even for a dry run to validate annotations and output fields.
        content = structure.render(atoms, deleted)
        report = {"input": str(args.input), "output": str(output), "dry_run": args.dry_run,
                  "input_atoms": len(atoms), "output_atoms": len(atoms) - len(deleted),
                  "surplus_o1_removed": len(removals), "atoms_removed": len(deleted),
                  "residues_reconstructed": len(renamed), "coordinates_changed": False,
                  "residue_changes": renamed, "removals": removals}
        print(f"{'DRY RUN' if args.dry_run else 'REPAIR'}: {len(removals)} surplus O1; "
              f"{len(deleted)} total atoms removed; {len(renamed)} sugar residues reconstructed.")
        for removal in removals:
            print(f"  Remove {removal['remove']}; C1 already links to {removal['link_to']}")
        if args.dry_run:
            print("No files written. Retained coordinates would be unchanged.")
            return 0
        with output.open("x") as handle:
            handle.write(content)
        if args.report:
            with args.report.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
        print(f"Wrote {output}; retained coordinates unchanged.")
        return 0
    except (RepairError, OSError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
