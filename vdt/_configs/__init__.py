"""
vdt._configs -- bundled YAML configuration files.

These configs are installed alongside the package so they are always
available after ``pip install vdt``, regardless of the working directory.

To resolve a bundled config path at runtime use importlib.resources::

    from importlib.resources import files
    cfg_path = files("vdt._configs").joinpath("default.yaml")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

Or use the convenience helper::

    from vdt._configs import get_config_path
    path = get_config_path("default")   # returns a Path object
"""
from __future__ import annotations
from importlib.resources import files
from pathlib import Path


def get_config_path(name: str) -> Path:
    """
    Return the filesystem path to a bundled config file.

    Parameters
    ----------
    name : str
        Config name without extension, e.g. ``"default"``.
        The function looks for ``<name>.yaml`` then ``<name>.yml``.

    Returns
    -------
    Path
        Absolute path to the config file, suitable for open() or yaml.safe_load().

    Raises
    ------
    FileNotFoundError
        If no matching file is found in the bundled configs.

    Examples
    --------
    >>> from vdt._configs import get_config_path
    >>> path = get_config_path("default")
    >>> import yaml
    >>> cfg = yaml.safe_load(path.read_text())
    """
    pkg = files("vdt._configs")
    for ext in (".yaml", ".yml"):
        candidate = pkg.joinpath(f"{name}{ext}")
        try:
            # traversable to concrete path
            return Path(str(candidate))
        except (TypeError, FileNotFoundError):
            continue
    raise FileNotFoundError(
        f"Bundled config '{name}' not found in vdt._configs. "
        f"Available configs: default"
    )
