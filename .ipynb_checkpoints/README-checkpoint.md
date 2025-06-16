# Scattering and scintillation analysis scripts for the DSA-110

## Scintillation Pipeline

scintillation/
│
├── data/
│   └── burst_power.npz         # Input data file
│
├── scint_analysis/
│   ├── configs/
│   │   ├── telescopes/
│   │   │   └── dsa.yaml        # Telescope-specific config file
│   │   └── bursts/
│   │       └── burst.yaml      # Burst-specific config
│   │
│   ├── scint_analysis/         # Pipeline source code
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── analysis.py
│   │   ├── config.py
│   │   ├── pipeline.py
│   │   └── plotting.py
│   │
│   └── run_analysis.py         # The script you will execute
│
└── cache/                      # Cache to preserve intermediate data products

These scripts contain helper functions that facilitate the analysis of burst properties within the PARSEC dashboard (see dsa110-pol repository).
