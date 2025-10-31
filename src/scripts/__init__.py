"""
Scripts and utilities for band structure and bulk evaluations.
Exposes DFTB+ runner helpers for external use.
"""

from .dftbp import (
    run_dftbp_bandgap,
    create_hsd_input,
    run_dftbp,
    parse_energy,
    parse_fermi,
    parse_band_out,
    calculate_bandgap,
    write_gen_format,
    plot_dos,
    save_band_data,
)

__all__ = [
    "run_dftbp_bandgap",
    "create_hsd_input",
    "run_dftbp",
    "parse_energy",
    "parse_fermi",
    "parse_band_out",
    "calculate_bandgap",
    "write_gen_format",
    "plot_dos",
    "save_band_data",
]
