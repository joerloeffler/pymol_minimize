"""PyMOL entry point for Quick Minimize."""

from .core import MinimizationError, minimize_selection, quick_relax, thorough_relax
from .diagnostics import check_installation

__all__ = [
    "MinimizationError", "check_installation", "minimize_selection",
    "quick_relax", "thorough_relax",
]


def __init_plugin__(app=None):
    """Register the Qt menu item and command when loaded as a PyMOL plugin."""
    from pymol import cmd
    from pymol.plugins import addmenuitemqt

    # Keep the original command name as a compatibility alias, but route it
    # through the same glycan-aware backend selection as the two-button UI.
    cmd.extend("quick_minimize", quick_relax)
    cmd.extend("quick_relax", quick_relax)
    cmd.extend("thorough_relax", thorough_relax)
    cmd.extend("quick_minimize_check", check_installation)
    cmd.extend("quick_minimize_gui", _show_dialog)
    addmenuitemqt("Quick Minimize", _show_dialog)


def _show_dialog():
    from .gui import show_dialog

    show_dialog()
