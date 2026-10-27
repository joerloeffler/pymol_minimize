#!/Users/joe/miniforge3/envs/pymol-quick-minimize/bin/python
"""Headless PDB relaxation using the pymol-quick-minimize environment.

Examples:
    ./minimize_pdb.py --pdb lig1_structure_1.pdb
    ./minimize_pdb.py --pdb model1.pdb model2.pdb --thorough
    ./minimize_pdb.py --pdb model.pdb --selection 'organic' --output ligand_relaxed.pdb

The full structure contributes to the force field. The default movable selection
is all atoms; the backend keeps its usual mild protein-backbone tether.
Ligands must already contain their complete explicit hydrogens. Their hydrogen
atoms are preserved, never added or removed. PDB bond orders/formal charges may
still require inference; use a chemically complete protonated input PDB.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


class PreserveLigandHydrogens:
    """Suppress PyMOL's nonpolymer H addition; retain OpenMM protein H prep."""

    def __init__(self, cmd):
        self.cmd = cmd

    def __getattr__(self, name):
        return getattr(self.cmd, name)

    def h_add(self, *args, **kwargs):
        return None


def ligand_signature(cmd):
    model = cmd.get_model('mol and not polymer and not solvent')
    atoms = tuple((a.index, a.name, a.resn, a.resi, a.chain, a.symbol,
                   a.formal_charge) for a in model.atom)
    hydrogen_bonds = tuple(sorted(
        tuple(sorted((model.atom[b.index[0]].index, model.atom[b.index[1]].index)))
        for b in model.bond
        if any(model.atom[i].symbol == 'H' for i in b.index)
    ))
    return atoms, hydrogen_bonds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdb', nargs='+', required=True, type=Path, help='One or more input PDBs')
    parser.add_argument('-o', '--output', type=Path, help='Output PDB (single input only)')
    parser.add_argument('--selection', default='all', help='Movable PyMOL selection (default: all)')
    parser.add_argument('--thorough', action='store_true', help='Use staged implicit-solvent relaxation')
    parser.add_argument('--iterations', type=int, help='Iteration limit (default: 100 quick / 1000 thorough)')
    args = parser.parse_args()
    if args.output and len(args.pdb) != 1:
        parser.error('--output requires exactly one --pdb input')
    if args.iterations is not None and args.iterations < 1:
        parser.error('--iterations must be positive')
    outputs = [args.output or p.with_name(p.stem + '_minimized.pdb') for p in args.pdb]
    input_paths = {p.resolve() for p in args.pdb}
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error('Inputs would produce duplicate output paths')
    for source, output in zip(args.pdb, outputs):
        if not source.is_file():
            parser.error(f'Input does not exist: {source}')
        if output.resolve() in input_paths or output.exists():
            parser.error(f'Refusing to overwrite an input or existing output: {output}')
        if not output.parent.is_dir():
            parser.error(f'Output directory does not exist: {output.parent}')

    try:
        import pymol2
        from pymol_quick_minimize import quick_relax, thorough_relax
        from pymol_quick_minimize.core import (
            _small_molecules_from_model, _without_spurious_hydrogen_bonds,
        )
        from pymol_quick_minimize.amber_backend import requires_compiled_backend
    except ImportError as exc:
        parser.exit(1, f'Missing dependency: {exc}\nUse the pymol-quick-minimize environment.\n')

    relax = thorough_relax if args.thorough else quick_relax
    options = {} if args.iterations is None else {'max_iterations': args.iterations}
    failures = 0
    for source, output in zip(args.pdb, outputs):
        try:
            with pymol2.PyMOL() as session:
                cmd = session.cmd
                cmd.load(str(source.resolve()), 'mol')
                if cmd.count_states('mol') != 1:
                    raise ValueError('Expected a single-model PDB; split multi-model files first')
                if not cmd.count_atoms('mol'):
                    raise ValueError('PDB contains no atoms')
                if not requires_compiled_backend(cmd, 'mol', 1):
                    molecules = _small_molecules_from_model(
                        _without_spurious_hydrogen_bonds(cmd.get_model('mol'))
                    )
                    for molecule in molecules:
                        if not any(a.atomic_number == 1 for a in molecule.atoms):
                            raise ValueError('Ligand has no explicit hydrogens. Supply a PDB '
                                             'with complete ligand hydrogens in the desired '
                                             'protonation state; this script will not guess them.')
                before = ligand_signature(cmd)
                result = relax(args.selection, 'mol', quiet=1,
                               _cmd=PreserveLigandHydrogens(cmd), **options)
                if ligand_signature(cmd) != before:
                    raise ValueError('Ligand atoms, formal charges, or hydrogen connectivity '
                                     'changed during preparation; refusing to save')
                # Exclusive creation protects previous results and input files.
                pdb_text = cmd.get_pdbstr('mol', state=1)
                with output.open('x') as handle:
                    handle.write(pdb_text)
                print(f'OK {source} -> {output}\n'
                      f'  Energy: {result.initial_energy_kj_mol:.2f} -> '
                      f'{result.final_energy_kj_mol:.2f} kJ/mol; '
                      f'{result.movable_atom_count}/{result.atom_count} prepared atoms movable', flush=True)
        except Exception as exc:
            failures += 1
            print(f'FAIL {source}: {exc}', file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
