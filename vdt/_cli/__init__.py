"""
vdt._cli -- console entry points installed by pip.

Three commands are registered in pyproject.toml::

    vdt-train   entry point -> vdt._cli.train:main
    vdt-bench   entry point -> vdt._cli.bench:main
    vdt-vis     entry point -> vdt._cli.vis:main

Each module is a thin shim that delegates to the corresponding top-level
script (train.py, benchmark.py, visualise.py) so that the scripts remain
usable both as direct ``python train.py`` invocations and as installed
console commands.

The shims accept the same CLI flags as the underlying scripts -- see
``vdt-train --help`` etc. after installation.
"""
