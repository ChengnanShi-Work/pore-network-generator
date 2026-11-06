# Stochastic full-scale pore network generator for ceramic brick

These Python research scripts generate stochastic pore networks for ceramic brick from networks extracted at six imaging scales. Each script fits and samples pore-body and pore-throat properties, constructs connections, and writes the generated network in Statoil format.

## Choose a domain size

| Script | Cubic domain edge | Domain volume |
| --- | ---: | ---: |
| `CB_FSPN500_improved_efficiency_20260602.py` | 0.5 mm | 0.125 mm³ |
| `CB_FSPN1000_improved_efficiency_20260602.py` | 1 mm | 1 mm³ |
| `CB_FSPN2000_improved_efficiency_20260602.py` | 2 mm | 8 mm³ |
| `CB_FSPN5000_improved_efficiency_20260602.py` | 5 mm | 125 mm³ |

The number immediately after FSPN in each filename is the domain edge in micrometres. The scripts also differ in the maximum pore-body and pore-throat radius bins used for fitting and generation. Use the matching script for each size.

## Required inputs

Each script loads three extracted Statoil networks at each of six imaging voxel sizes: 4.8, 2.4, 1.2, 0.6 and 0.3 µm, and 128 nm. Supply all 18 network prefixes, each with its matching `_node1.dat`, `_node2.dat`, `_link1.dat` and `_link2.dat` files. Some prefixes contain the literal suffix `.raw`; these are network basenames. The generator reads extracted networks rather than XCT/XRM image volumes.

The scripts use pore-body and pore-throat radii, volumes and shape factors from the imported networks. Radius-bin assignments and target porosity contributions are set in the script for this ceramic brick case. The four scripts require these input files to be supplied separately.

## Run the smallest case

1. Use the original Python environment if available. The scripts import NumPy, pandas, SciPy, Matplotlib, OpenPNM, copulas and sobol_seq. The OpenPNM version must provide `op.io.Statoil.load`. Exact package versions have not yet been verified.
2. In the 500 script, update the input directories and prefixes in the six loading blocks. Check each prefix against its actual filenames. Set `folder` to an existing output directory. The original path construction uses Windows separators.
3. From the directory containing the script, run:

   ```sh
   python CB_FSPN500_improved_efficiency_20260602.py
   ```

The script writes four files per realization using the generated network prefix: `_node1.dat`, `_node2.dat`, `_link1.dat` and `_link2.dat`. Use an empty output directory to avoid overwriting earlier runs. These instructions are based on source inspection; an end-to-end run requires the extracted input networks.

## Associated publication

Shi, C. and Janssen, H. (2026). Stochastic full-scale pore network generation for porous building materials: case study on ceramic brick. *NDT & E International*, 161, 103709. [Article DOI](https://doi.org/10.1016/j.ndteint.2026.103709)
