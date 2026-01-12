# LLAMAS Quicklook

Quick look tools for LLAMAS spectroscopic data reduction and analysis.

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate llamas-quicklook
```

## Project Structure

- `utils/` - Core analysis modules
  - `io.py` - FITS file I/O for LLAMAS spectra
  - `geometry.py` - Fiber mapping and 2D Gaussian fitting
  - `whitelight.py` - Whitelight image construction
  - `coadd.py` - Weighted spectral coadding
  - `plotting.py` - Visualization tools
  - `paths.py` - Path management utilities

- `notebooks/` - Analysis notebooks
  - `1_io.ipynb` - Basic I/O examples
  - `2_rss_to_1dspec.ipynb` - RSS to 1D spectrum extraction pipeline

- `config/` - Configuration files
  - `LLAMAS_FiberMap_rev04.dat` - Fiber position lookup table

## Usage

See the notebooks in `notebooks/` for example workflows.

## Dependencies

- numpy
- astropy
- matplotlib
- scipy
- seaborn
