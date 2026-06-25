"""
vdeductive._cli.vis -- console entry point for ``vdeductive-vis``.

Thin shim around the top-level visualise.py script.  After
``pip install vdeductive`` run::

    vdeductive-vis --help

All CLI flags are identical to visualise.py.  The script is resolved
relative to the installed package root so it works from any directory.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path


def main() -> None:
    """
    Entry point for the ``vdeductive-vis`` console command.

    Locates visualise.py at the repository root and executes it in the
    current process via runpy so that argparse reads sys.argv.
    """
    _root = Path(__file__).resolve().parent.parent.parent
    _script = _root / "visualise.py"
    if not _script.exists():
        print(
            "[vdeductive-vis] ERROR: could not locate visualise.py at:\n"
            f"  {_script}\n"
            "Make sure you installed vdeductive from the repository root with:\n"
            "  pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    runpy.run_path(str(_script), run_name="__main__")


if __name__ == "__main__":
    main()
