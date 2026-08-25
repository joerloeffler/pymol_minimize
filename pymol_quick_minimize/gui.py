"""Small Qt front end for the quick_minimize PyMOL command."""

from __future__ import annotations

_dialog = None


def show_dialog():
    from pymol import cmd
    from pymol.Qt import QtWidgets

    global _dialog
    if _dialog is not None:
        _dialog.show()
        _dialog.raise_()
        return

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Quick Minimize")
    layout = QtWidgets.QFormLayout(dialog)

    movable = QtWidgets.QLineEdit("sele")
    system = QtWidgets.QLineEdit("all")
    iterations = QtWidgets.QSpinBox()
    iterations.setRange(0, 100000)
    iterations.setValue(0)
    iterations.setSpecialValueText("Auto (100 quick / 1000 thorough)")
    restraint = QtWidgets.QDoubleSpinBox()
    restraint.setRange(0, 100000)
    restraint.setDecimals(1)
    restraint.setValue(1000.0)
    hydrogens = QtWidgets.QCheckBox("Add missing hydrogens")
    hydrogens.setChecked(True)
    from .diagnostics import check_installation

    readiness = check_installation(quiet=1)
    status = QtWidgets.QLabel(readiness.summary)
    status.setWordWrap(True)

    layout.addRow("Movable selection", movable)
    layout.addRow("Complete system", system)
    layout.addRow("Maximum iterations", iterations)
    layout.addRow("Outside restraint (kJ/mol/nm²)", restraint)
    layout.addRow(hydrogens)
    layout.addRow(status)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    quick_button = buttons.addButton("Quick Relax", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
    thorough_button = buttons.addButton(
        "Thorough Relax", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
    )
    layout.addRow(buttons)
    buttons.rejected.connect(dialog.close)

    def run(mode):
        from .core import quick_relax, thorough_relax

        try:
            status.setText("Minimizing…")
            QtWidgets.QApplication.processEvents()
            relax = quick_relax if mode == "quick" else thorough_relax
            options = {
                "restraint": restraint.value(),
                "add_hydrogens": hydrogens.isChecked(),
                "_cmd": cmd,
            }
            if iterations.value():
                options["max_iterations"] = iterations.value()
            result = relax(movable.text(), system.text(), **options)
            status.setText(
                f"Done: relaxed {result.movable_atom_count} atoms "
                f"({result.atom_count} in the complete system)."
            )
        except Exception as exc:
            status.setText(str(exc))
            QtWidgets.QMessageBox.critical(dialog, "Quick Minimize", str(exc))

    quick_button.clicked.connect(lambda: run("quick"))
    thorough_button.clicked.connect(lambda: run("thorough"))
    quick_button.setEnabled(readiness.ok)
    thorough_button.setEnabled(readiness.ok)
    if not readiness.ok:
        status.setToolTip("\n".join(readiness.details))
    _dialog = dialog
    dialog.show()
