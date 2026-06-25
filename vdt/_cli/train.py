"""
vdt._cli.train -- console entry point for ``vdt-train``.

This is a thin shim around the top-level train.py script.  It exists so
that after ``pip install vdt`` users can run::

    vdt-train --config configs/mps.yaml --dataset cora

instead of::

    python train.py --config configs/mps.yaml --dataset cora

All CLI flags are identical to those accepted by train.py.  See
``vdt-train --help`` for the full list.

The shim resolves the train.py path relative to the installed package
root, so it works correctly from any working directory.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path


def main() -> None:
    """
    Entry point for the ``vdt-train`` console command.

    Locates train.py at the repository / install root (one level above the
    vdt package directory) and executes it in the current process via
    runpy.run_path so that argparse reads sys.argv as normal.
    """
    _root = Path(__file__).resolve().parent.parent.parent
    _train_script = _root / "train.py"
    if not _train_script.exists():
        print(
            "[vdt-train] ERROR: could not locate train.py at expected path:\n"
            f"  {_train_script}\n"
            "Make sure you installed vdt from the repository root with:\n"
            "  pip install -e .",
            file=sys.stderr,
        )
        sys.exit(1)
    runpy.run_path(str(_train_script), run_name="__main__")


if __name__ == "__main__":
    main()
