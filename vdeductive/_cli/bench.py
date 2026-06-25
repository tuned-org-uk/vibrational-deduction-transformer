"""
vdeductive._cli.bench -- console entry point for ``vdeductive-bench``.

Thin shim around the top-level benchmark.py script.  After
``pip install vdeductive`` run::

    vdeductive-bench --help

All CLI flags are identical to benchmark.py.  The script is resolved
relative to the installed package root so it works from any directory.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path


def main() -> None:
    """
    Entry point for the ``vdeductive-bench`` console command.

    Locates benchmark.py at the repository / install root and executes it
    in the current process via runpy so that argparse reads sys.argv.
    """
    _root = Path(__file__).resolve().parent.parent.parent
    _script = _root / "benchmark.py"
    if not _script.exists():
        print(
            "[vdeductive-bench] ERROR: could not locate benchmark.py at:\n"
            f"  {_script}\n"
            "Make sure you installed vdeductive from the repository root with:\n"
            "  pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    runpy.run_path(str(_script), run_name="__main__")


if __name__ == "__main__":
    main()
