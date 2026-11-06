# Single-scale pore network generators (G₄–T₂)

These material-specific Python scripts implement geometry algorithm G₄ and topology algorithm T₂ to generate stochastic networks using geometry statistics and coordination numbers from an extracted network. They sample correlated pore-body and pore-throat properties, adjust volume contributions in pore-radius bins, assign coordinates and connections, and export generated networks in Statoil format.

## Scripts and current settings

The following values are read directly from the uploaded scripts. Domain dimensions are calculated as dim_new multiplied by res.

| Script | Material | Input prefix | dim_new | res (µm) | Generated cubic edge (mm) | Test |
| --- | --- | --- | --- | ---: | ---: | ---: |
| KC_G4_T2.py | Ketton carbonate | Ketton_500 | [500, 500, 500] | 3.00006 | 1.50003 | 10 |
| BS_G4_T2.py | Berea sandstone | Berea | [400, 400, 400] | 5.345 | 2.138 | 10 |
| SG_G4_T2.py | Sintered glass | sintered_glass | [500, 500, 500] | 2.5 | 1.25 | 1 |
| CB_G4_T2.py | Ceramic brick | S3_3_1000 | [1000, 1000, 1000] | 2.4 | 2.4 | 1 |

Test controls the number of generated realizations. For CB, dim_new is [1000, 1000, 1000] while dim for the original image volume is [500, 500, 500]. Check the latter against the actual extracted input volume.

## Inputs and configuration

Each script reads one extracted network using OpenPNM's op.io.Statoil.load. Supply the four matching files for its prefix: _node1.dat, _node2.dat, _link1.dat and _link2.dat.

The code reads pore-body and pore-throat radius, volume and shape factor. Coordination numbers are calculated from the imported connections.

Update the input path and prefix, and set folder to an existing output directory. The current file-writing code uses Windows path separators. Verify dim against the input image dimensions, res against its voxel size, and dim_new against the intended generated domain.

The scripts contain material-specific fitted distributions in result_of_dist and target porosity contributions in target_porosity_bin and target_porosity_bin_pt. Automatic fitting calls are commented out. Changing the input path alone is insufficient to configure a different material. The CB implementation also uses log-transformed radius and volume distributions.

## Environment and execution

Imported third-party packages include NumPy, SciPy, Matplotlib, OpenPNM, seaborn and sobol_seq. Exact versions remain to be verified against the original execution environment. OpenPNM must provide op.io.Statoil.load.

After configuring the inputs and output directory, run from this folder, for example:

```sh
python KC_G4_T2.py
```

Each realization writes _node1.dat, _node2.dat, _link1.dat and _link2.dat files. Use an empty output directory for a new run.

## Version status

These scripts document the currently available material-specific implementations. Final parameter settings used for the associated publication remain to be confirmed. They have been inspected as source code; end-to-end execution requires the extracted input networks and a compatible environment.

## Associated publication

Shi, C. and Janssen, H. (2025). Improved Algorithms for Stochastic Pore Network Generation for Porous Materials. Transport in Porous Media, 152, article 33.
