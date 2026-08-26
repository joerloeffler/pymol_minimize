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
