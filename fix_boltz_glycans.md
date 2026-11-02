# Local glycan repair

Run `fix_boltz_glycans.py` in the `pymol-quick-minimize` conda environment.
It reads PDB or mmCIF locally. No network services are used.

From the project directory:

```bash
conda activate pymol-quick-minimize
python dev/fix_boltz_glycans.py dev/updated_issues/input_model_0.cif --dry-run
python dev/fix_boltz_glycans.py model.cif --report model_repair.json
```

The second command writes `model_glycan_fixed.cif`. Existing files, inputs,
and reference files are never overwritten. `--output` selects another output
name in the same format. `--dry-run` writes nothing, including no report.

The script checks pyranose ring connectivity and direct C1 linkages before
removing surplus O1 atoms. It also removes hydrogens attached exclusively to
those deleted O1 atoms. Free anomeric hydroxyls and O1-mediated bridges remain.
The inferred direct linkage must join C1 to another supported sugar's
O2/O3/O4/O6 or ASN/ND2. Remaining coordinates, protein residues, and AIB are
unchanged. It does not minimize the structure.

These checks assume the direct, covalent-length glycosidic contacts are intended.
They do not establish that a predicted glycan has the correct biological topology.
Use the dry run and optional JSON report to review deletions and bond distances.
Unrecognized links, alternate conformations, incomplete rings, explicit link
annotations that would need updating, and ambiguous bonds are rejected.

For merged sugar residues, an optional reference supplies the intended residue
names and numbering:

```bash
python dev/fix_boltz_glycans.py merged.cif --reference reference.cif --dry-run
```

The reference must have matching sugar chains, atom order/inventories, linkages,
and stereocenters, including C1 anomers. Its coordinates are not copied. The
script does not guess MAN versus GLC, nor change an alpha/beta assignment to
accommodate distorted geometry. The supplied newer model differs from the
original at three C1 anomers, so that particular reference pairing is rejected.

Local validation on `input_model_0.cif` removes 33 surplus O1 atoms. The repaired
copy passes Quick and Thorough AIB relaxation, and running the repair again
removes no additional atoms. Originals remain unchanged.

Run the synthetic safety checks with:

```bash
python -m unittest discover -s dev -p test_fix_boltz_glycans.py -v
```
