# PyMOL Quick Minimize

Two-button, MOE-like geometry cleanup inside PyMOL. The complete structure is
included in the force-field calculation, while only the chosen pocket, ligand,
glycan, side chains, or whole model is allowed to relax freely.

- **Quick Relax** is the everyday button: 100 L-BFGS iterations in a fast
  nonperiodic field.
- **Thorough Relax** uses staged restraints, 1,000 final iterations, and OBC2
  implicit solvent. Use it for difficult clashes or after model building.

This is a geometry-repair tool, not an energy-scoring workflow. It preserves the
protein backbone with a mild tether and restrains everything outside the movable
selection.

## Installation

### macOS and Linux

Use a new Miniforge environment. PyMOL and the chemistry stack must use the same
Python installation; adding the plugin ZIP to an unrelated PyMOL installation is
not enough.

```bash
./install.sh
conda activate pymol-quick-minimize
pymol
```

The equivalent manual command is `conda env create -f environment.yml`. On
Apple Silicon, install the arm64 build of Miniforge; do not run an Intel conda
installation through Rosetta unless dependency solving specifically requires it.

Then, in that copy of PyMOL:

1. Open **Plugin → Plugin Manager → Install New Plugin**.
2. Install `pymol-quick-minimize.zip` from this directory.
3. Restart PyMOL.
4. Enter `quick_minimize_check` in the PyMOL command line.

The last command must say `Quick Minimize is ready.` and show successful checks
for OpenMM, OpenFF Sage, RDKit, antechamber, cpptraj, and tleap.

The environment is the reproducible distribution unit. It includes PyMOL,
OpenMM, OpenMMForceFields, OpenFF Toolkit/Sage, PDBFixer, RDKit, and AmberTools.
The plugin itself has no hidden pip downloads.

### Windows

Use WSL2 with Ubuntu and WSLg. Native Windows is not currently supported because
the complete AmberTools/OpenFF/PyMOL environment is not available and tested as
one native Windows stack.

In an Administrator PowerShell:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Restart Windows if requested, open the Ubuntu application, install Miniforge for
Linux inside Ubuntu, and clone or copy this project into the WSL filesystem. Then
run the same Linux commands:

```bash
./install.sh
conda activate pymol-quick-minimize
pymol
```

PyMOL should appear as a Windows desktop window through WSLg. Install the plugin
ZIP from inside PyMOL and run `quick_minimize_check` exactly as above. Files on
the Windows C: drive are visible under `/mnt/c/`, although keeping the environment
and project in the Linux home directory generally avoids filesystem overhead.

This WSL route is the practical Windows installation, but it still needs a clean
Windows CI/manual release test before being labelled fully supported. See
[Microsoft's WSL GUI application guide](https://learn.microsoft.com/windows/wsl/tutorials/gui-apps)
if the PyMOL window does not appear.

A separately downloaded Incentive/commercial PyMOL is not currently the easy
path on any platform because its private Python environment may not see the conda
packages.

## Use

Open **Plugin → Quick Minimize**, enter the movable selection and complete
system, and click either button. `Auto` iterations means 100 for Quick and the
full staged protocol for Thorough.

If the menu entry is hidden by a PyMOL/plugin-manager issue, enter
`quick_minimize_gui` to open the same two-button window. The compatibility command
`quick_minimize` is an alias for `quick_relax`. With no active `sele`, either
command relaxes the complete system instead of reporting an invalid selection.

For a ligand pocket:

```pml
select pocket, byres (polymer within 5 of organic) or organic
quick_relax pocket, all
thorough_relax pocket, all
```

For the complete visible model: `quick_relax all, all`.

The first argument is movable; the second must be a complete molecular system.
Do not use a cut-out protein fragment as the system. The default outside restraint
is 1,000 kJ/mol/nm². The CPU OpenMM platform is used for predictable behavior on
ordinary workstations.

## Force-field routing

The plugin chooses a backend from the structure:

- Standard proteins/nucleic acids use Amber ff14SB in OpenMM.
- Disconnected drug-like ligands use OpenFF Sage 2.2.1 with AM1-BCC charges.
  Parameters are cached, so the first run can take about a minute and repeats are
  much faster.
- Recognized covalent glycans use GLYCAM06j with ff14SB, compiled automatically
  by AmberTools and minimized by OpenMM.
- AIB is supported using the Amber ff15ipq-m template plus the packaged bridge
  parameter needed at an ff14SB peptide bond. AIB relaxation adds targeted
  geometry guards for its gem-dimethyl angle and both adjoining peptide-bond
  planarity terms. A post-relaxation check rejects non-planar peptide bonds or a
  still-collapsed `CB1-CA-CB2` angle instead of returning bad coordinates.

For ambiguous PDB/mmCIF ligands, the plugin attempts bond-order and common net
charge inference. SDF or MOL2 with explicit bond orders and formal charges is
still preferred. Inferred parameters are suitable for clash cleanup, not a claim
of thermodynamic accuracy.

Current deliberate limitations include metal coordination, covalent inhibitors,
radicals, arbitrary modified residues, and mixed glycan systems containing an
additional unsupported ligand. These fail with an error instead of silently
inventing a topology. Thorough mode's OBC2/GLYCAM combination is a pragmatic
geometry solvent model and may emit a non-optimal GB-radius warning.

Protein heavy atoms and termini are repaired in the OpenMM working model with
PDBFixer; missing residues are never invented. Protein hydrogens are temporary.
Ligand hydrogens are added to the PyMOL object. Linked PDB glycans that retain an
anomeric O1 despite an inter-residue C1 bond are rejected with a residue-specific
error. The plugin never silently deletes these chemically impossible atoms.

## Validation included here

Both supplied development structures run through the public plugin commands:

- `input_model_0.pdb` is now the negative control: its 33 surplus anomeric O1
  atoms produce a clear error, and the PyMOL object remains unchanged.
- `san_glycan.pdb`: the cleaned protein, three N-glycans, and AIB; 5,533 prepared
  atoms. Quick Relax completes in roughly 4.7 seconds on the development Mac.
- `PET_test.cif`: protein plus a ligand; 7,807 prepared atoms and 464 movable
  pocket atoms. Cached Quick Relax takes roughly 6 seconds; Thorough Relax
  roughly 70 seconds.

Using a simple heavy-atom overlap audit that excludes bonded and 1–3 pairs, severe
overlaps (>0.6 Å van der Waals overlap) changed from 96 to 9 for the cleaned glycoprotein
and from 162 to 19 for the protein–ligand model after Quick Relax. Remaining cases
were marginal (worst approximately 0.64 Å), not catastrophic contacts.

## Development checks

```bash
python -m compileall -q pymol_quick_minimize
python -m unittest discover -s tests -v
```

## Headless AIB batch relaxation

Do not rely on GUI plugin command registration in `pymol -c`; some PyMOL builds
skip plugin autoloading in headless mode. The included runner imports the same
backend explicitly and returns a nonzero exit status if any file fails:

```bash
conda activate pymol-quick-minimize
python batch_relax_aib.py ./*.pdb --output-dir minimized
```

Each input is loaded into a fresh PyMOL session, only `resn AIB` is movable, and
the complete structure remains in the force-field system. Outputs are named
`<input>_AIB_relaxed.pdb`; files with no AIB or invalid connectivity are reported
as failures rather than silently copied.

Release-quality cross-platform distribution still requires clean macOS, Linux,
and eventually Windows CI runs rather than assuming that one working machine is
enough.
