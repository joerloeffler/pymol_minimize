"""Scientific smoke tests; run inside the environment.yml environment."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pymol
from pymol import cmd

from pymol_quick_minimize import check_installation, minimize_selection, quick_relax


class QuickMinimizeSmokeTests(unittest.TestCase):
    def setUp(self):
        cmd.reinitialize()

    def test_dependencies_are_complete(self):
        self.assertTrue(check_installation(quiet=1).ok)

    def test_openff_ligand_and_multiple_objects(self):
        cmd.fragment("benzene", "lig")
        cmd.fragment("methane", "cosolvent")
        with tempfile.TemporaryDirectory() as temp:
            result = minimize_selection(
                "lig", "all", max_iterations=2, restraint=1000, quiet=1,
                cache=str(Path(temp) / "templates.json"), _cmd=cmd,
            )
        self.assertEqual(result.movable_atom_count, 12)
        self.assertTrue(np.isfinite(cmd.get_coords("all")).all())

    def test_protein_hydrogen_insertion_round_trip(self):
        fixture = Path(pymol.__file__).parent / "pymol_path/test/dat/pept.pdb"
        if not fixture.exists():
            self.skipTest("PyMOL test peptide is not included in this build")
        cmd.load(str(fixture), "pep")
        cmd.remove("hydro")
        before = cmd.get_coords("pep").copy()
        with tempfile.TemporaryDirectory() as temp:
            result = minimize_selection(
                "pep and resi 1", "pep", max_iterations=2, quiet=1,
                cache=str(Path(temp) / "templates.json"), _cmd=cmd,
            )
        after = cmd.get_coords("pep")
        self.assertEqual(before.shape, after.shape)
        self.assertTrue(np.isfinite(after).all())
        self.assertLess(np.linalg.norm(after - before, axis=1).max(), 5.0)
        self.assertLess(result.final_energy_kj_mol, result.initial_energy_kj_mol)

    def test_interleaved_protein_and_water_round_trip(self):
        fixture = Path(__file__).resolve().parents[1] / "dev/test_me.pdb"
        if not fixture.exists():
            self.skipTest("Development structure is not included")
        cmd.load(str(fixture), "pep")
        cmd.remove("resn UNK")
        before = cmd.get_coords("pep").copy()
        result = quick_relax(
            "pep and polymer and resi 1", "pep", max_iterations=2,
            quiet=1, cache="", _cmd=cmd,
        )
        after = cmd.get_coords("pep")
        self.assertEqual(before.shape, after.shape)
        self.assertTrue(np.isfinite(after).all())
        self.assertLess(np.linalg.norm(after - before, axis=1).max(), 5.0)
        self.assertLess(result.final_energy_kj_mol, result.initial_energy_kj_mol)

    def test_extra_hydrogen_bond_and_repeated_ligand_names(self):
        from pymol_quick_minimize.core import _build_topology, _without_spurious_hydrogen_bonds
        cmd.fragment("methane", "lig")
        cmd.fragment("methane", "other")
        cmd.alter("other", "resi='2'")
        cmd.create("combined", "lig or other")
        cmd.delete("lig")
        cmd.delete("other")
        cmd.bond("combined and resi 1 and elem H and index 2", "combined and resi 2 and elem C")
        model = cmd.get_model("all")
        cleaned = _without_spurious_hydrogen_bonds(model)
        self.assertEqual(len(model.bond) - len(cleaned.bond), 1)
        cmd.alter("combined and resi 1 and elem H", "name='H'")
        model = cmd.get_model("combined and resi 1")
        topology = _build_topology(model)
        self.assertEqual(len({a.name for a in topology.atoms()}), len(model.atom))
        self.assertEqual([a.id for a in topology.atoms()], [str(i + 1) for i in range(len(model.atom))])

    def test_delocalized_carboxylate_and_guanidinium(self):
        from types import SimpleNamespace
        from rdkit import Chem
        from openff.units import unit
        from pymol_quick_minimize.core import _small_molecules_from_model

        for smiles, expected_charge in (("CC(=O)[O-]", -1), ("NC(N)=[NH2+]", 1)):
            molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = [
                SimpleNamespace(symbol=a.GetSymbol(), elem=a.GetSymbol(),
                                formal_charge=0, name=f"A{a.GetIdx()}",
                                resn="LIG", coord=[0.0, 0.0, 0.0])
                for a in molecule.GetAtoms()
            ]
            bonds = []
            for bond in molecule.GetBonds():
                left, right = bond.GetBeginAtom(), bond.GetEndAtom()
                resonance = any(
                    a.GetSymbol() == "C" and a.GetDegree() == 3
                    and sum(n.GetSymbol() == "O" for n in a.GetNeighbors()) == 2
                    and other.GetSymbol() == "O"
                    or a.GetSymbol() == "C" and a.GetDegree() == 3
                    and all(n.GetSymbol() == "N" for n in a.GetNeighbors())
                    for a, other in ((left, right), (right, left))
                )
                bonds.append(SimpleNamespace(
                    index=[left.GetIdx(), right.GetIdx()],
                    order=4 if resonance else int(bond.GetBondTypeAsDouble()),
                ))
            converted = _small_molecules_from_model(SimpleNamespace(atom=atoms, bond=bonds))
            self.assertEqual(len(converted), 1)
            self.assertEqual(converted[0].n_atoms, len(atoms))
            self.assertEqual(converted[0].total_charge.m_as(unit.elementary_charge), expected_charge)

    def test_terminal_repair_preserves_unknown_ligand_bonds(self):
        from openmm import Vec3, unit
        from openmm.app import Element, Topology
        from pymol_quick_minimize.core import _repair_missing_heavy_atoms, _topology_atom_key

        topology = Topology()
        protein = topology.addResidue("GLY", topology.addChain("A"))
        positions = []
        for name, element, xyz in (
            ("N", "N", (0, 0, 0)), ("CA", "C", (0.145, 0, 0)),
            ("C", "C", (0.2, 0.14, 0)), ("O", "O", (0.16, 0.25, 0)),
        ):
            topology.addAtom(name, Element.getBySymbol(element), protein)
            positions.append(Vec3(*xyz))
        ligand = topology.addResidue("UNK", topology.addChain("B"))
        carbon = topology.addAtom("C1", Element.getBySymbol("C"), ligand)
        positions.append(Vec3(2, 0, 0))
        for i, xyz in enumerate(((2.1, 0, 0), (2, 0.1, 0), (2, 0, 0.1), (1.9, -0.1, -0.1)), 1):
            hydrogen = topology.addAtom(f"HX{i}", Element.getBySymbol("H"), ligand)
            positions.append(Vec3(*xyz))
            topology.addBond(carbon, hydrogen, order=1)
        topology.createStandardBonds()
        def ligand_edges(topo):
            return {
                frozenset((_topology_atom_key(left), _topology_atom_key(right)))
                for left, right in topo.bonds()
                if left.residue.name == "UNK" or right.residue.name == "UNK"
            }
        expected = ligand_edges(topology)
        repaired, repaired_positions = _repair_missing_heavy_atoms(topology, positions * unit.nanometer)
        self.assertIn("OXT", [a.name for a in next(repaired.residues()).atoms()])
        self.assertEqual(ligand_edges(repaired), expected)
        self.assertEqual(len(expected), 4)
        self.assertTrue(np.isfinite(repaired_positions.value_in_unit(unit.nanometer)).all())

    def test_calcium_uses_amber_charge(self):
        from openmm import NonbondedForce, unit
        from openmm.app import ForceField
        import openmmforcefields  # Register the Amber force-field search path.
        from pymol_quick_minimize.core import _build_topology, _small_molecules_from_model

        cmd.pseudoatom("ion", name="CA", resn="CA", elem="Ca")
        model = cmd.get_model("ion")
        self.assertEqual(_small_molecules_from_model(model), [])
        system = ForceField("amber/tip3p_standard.xml").createSystem(_build_topology(model))
        force = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
        charge, _, _ = force.getParticleParameters(0)
        self.assertEqual(charge.value_in_unit(unit.elementary_charge), 2.0)
        cmd.alter("ion", "formal_charge=1")
        from pymol_quick_minimize import MinimizationError
        with self.assertRaisesRegex(MinimizationError, "Unsupported charge"):
            _small_molecules_from_model(cmd.get_model("ion"))
        cmd.alter("ion", "formal_charge=0")
        result = quick_relax("ion", "ion", max_iterations=2, quiet=1, cache="", _cmd=cmd)
        self.assertTrue(np.isfinite(result.final_energy_kj_mol))

    def test_quick_relax_without_named_selection_uses_complete_system(self):
        cmd.fragment("benzene", "lig")
        self.assertNotIn("sele", cmd.get_names("selections"))
        with tempfile.TemporaryDirectory() as temp:
            result = quick_relax(
                system="all", max_iterations=2, quiet=1,
                cache=str(Path(temp) / "templates.json"), _cmd=cmd,
            )
        self.assertEqual(result.movable_atom_count, result.atom_count)

    def test_pymol_self_keyword_reaches_small_molecule_backend_safely(self):
        cmd.fragment("benzene", "lig")
        with tempfile.TemporaryDirectory() as temp:
            result = quick_relax(
                "all", "all", max_iterations=2, quiet=1,
                cache=str(Path(temp) / "templates.json"), _self=cmd,
            )
        self.assertEqual(result.movable_atom_count, result.atom_count)


if __name__ == "__main__":
    unittest.main()
