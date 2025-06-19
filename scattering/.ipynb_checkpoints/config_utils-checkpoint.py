"""
config_utils.py
===============

Utility for reading telescope-specific raw-data parameters from
*telescopes.yaml*.  Keeping this separate avoids a hard dependency on
`pyyaml` in the core physics modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml


def load_telescope(
    telescope: str,
    cfg_path: str | Path = "telescopes.yaml",
) -> Dict[str, float]:
    """
    Return the raw instrumental constants for *telescope*.

    Parameters
    ----------
    telescope
        Key name in the YAML file (e.g. ``"DSA-110"``).
    cfg_path
        Path to the YAML configuration file; default = project root.

    Returns
    -------
    dict
        ``{"df_MHz_raw": …, "dt_ms_raw": …, "f_min_GHz": …, "f_max_GHz": …}``

    Raises
    ------
    FileNotFoundError
        If *cfg_path* does not exist.
    KeyError
        If *telescope* is not found in the YAML file.
    ValueError
        If any of the four required fields is missing or ``null``.
    """
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file {cfg_path} not found")

    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    if telescope not in cfg:
        raise KeyError(f"Telescope '{telescope}' not in {cfg_path}")

    required = ("df_MHz_raw", "dt_ms_raw", "f_min_GHz", "f_max_GHz")
    entry = cfg[telescope]

    missing = [k for k in required if k not in entry or entry[k] is None]
    if missing:
        raise ValueError(
            f"Telescope '{telescope}' missing fields {missing} "
            f"in {cfg_path}"
        )

    return {k: float(entry[k]) for k in required}
