# Stochastic pore network generator

Python research scripts developed at KU Leuven for stochastic generation of single-scale and full-scale pore networks from image-derived network information.

## Code

- [Single-scale generators](single-scale/README.md) implement geometry algorithm G₄ with topology algorithm T₂ for Ketton carbonate, Berea sandstone, sintered glass and ceramic brick.
- [Full-scale generators](full-scale/README.md) combine multiscale pore statistics to generate ceramic brick networks for four cubic domain sizes.

The scripts require extracted pore networks in Statoil format as input. Each folder documents its inputs, configuration and outputs.

## Publications

Shi, C. and Janssen, H. (2025). Improved Algorithms for Stochastic Pore Network Generation for Porous Materials. *Transport in Porous Media*, 152, article 33.

Shi, C. and Janssen, H. (2026). Stochastic full-scale pore network generation for porous building materials: case study on ceramic brick. *NDT & E International*, 161, 103709. [Article DOI](https://doi.org/10.1016/j.ndteint.2026.103709)

## Status

The single-scale scripts implement G₄–T₂ with material-specific settings. Their final parameter configurations for the 2025 publication are being checked. Runtime environment versions and end-to-end execution with the required input data are also pending verification.
