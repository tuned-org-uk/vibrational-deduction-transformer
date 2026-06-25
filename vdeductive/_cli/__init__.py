"""
vdeductive._cli -- console entry points installed by pip.

Three commands are registered in pyproject.toml::

    vdeductive-train   entry point -> vdeductive._cli.train:main
    vdeductive-bench   entry point -> vdeductive._cli.bench:main
    vdeductive-vis     entry point -> vdeductive._cli.vis:main

Each module is a thin shim that delegates to the corresponding top-level
script (train.py, benchmark.py, visualise.py) so that the scripts remain
usable both as direct ``python train.py`` invocations and as installed
console commands.

The shims accept the same CLI flags as the underlying scripts -- see
``vdeductive-train --help`` etc. after installation.
"""
