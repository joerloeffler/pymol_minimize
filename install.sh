#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v mamba >/dev/null 2>&1; then
    environment_manager="mamba"
elif command -v conda >/dev/null 2>&1; then
    environment_manager="conda"
else
    echo "Miniforge/conda was not found. Install Miniforge first:" >&2
    echo "https://github.com/conda-forge/miniforge" >&2
    exit 1
fi

"$environment_manager" env create --file "$project_dir/environment.yml"

echo
echo "Installation complete."
echo "1. Run: conda activate pymol-quick-minimize"
echo "2. Run: pymol"
echo "3. Install $project_dir/pymol-quick-minimize.zip with PyMOL's Plugin Manager."
echo "4. Restart PyMOL and run: quick_minimize_check"

