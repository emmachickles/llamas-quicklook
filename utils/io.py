import numpy as np
from astropy.io import fits
from astropy.table import Table
from pathlib import Path
import logging
import os

def load_llamas_spectrum(filepath):
    """
    Load a reduced 1D LLAMAS spectrum from FITS file.
    Returns dict with wave/flux/error/mask/fwhm/fibermap.
    """
    hdul = fits.open(filepath)

    flux = hdul[1].data
    error = hdul[2].data
    mask = hdul[3].data
    wave = hdul[4].data
    fwhm = hdul[5].data
    fibermap = hdul[6].data

    hdul.close()

    return {
        "wave": wave,
        "flux": flux,
        "error": error,
        "mask": mask,
        "fwhm": fwhm,
        "fibermap": fibermap,
        "n_fibers": len(fibermap),
    }


def extract_fiber_spectrum(f_spec, fiber_index):
    """Extract (wave, flux) for one fiber index from reduced spectrum."""
    spec_data = load_llamas_spectrum(f_spec)
    wave = spec_data["wave"][fiber_index]
    flux = spec_data["flux"][fiber_index]
    return wave, flux


def save_llamas_spectrum(coadded, exposure, output_dir):
    """
    Save extracted LLAMAS spectrum to FITS and ASCII files.
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = exposure["base_name"]

    for band in ["blue", "green", "red"]:
        if coadded.get(band) is None:
            continue

        spec = coadded[band]

        fits_path = output_dir / f"{base_name}_{band}_coadd.fits"
        hdu_list = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(spec["wave"], name="WAVE"),
            fits.ImageHDU(spec["flux"], name="FLUX"),
            fits.ImageHDU(spec["error"], name="ERROR"),
            fits.ImageHDU(spec.get("obj_flux", np.array([])), name="OBJ_FLUX"),
            fits.ImageHDU(spec.get("sky_flux", np.array([])), name="SKY_FLUX"),
        ])

        hdu_list[0].header["OBJECT"] = exposure.get("base_name", "")
        hdu_list[0].header["MJD-OBS"] = exposure.get("mjd_obs", -1.0)
        hdu_list[0].header["EXPTIME"] = exposure.get("exptime", -1.0)
        hdu_list[0].header["BAND"] = band
        if "n_obj_fibers" in spec:
            hdu_list[0].header["NOBJ"] = spec["n_obj_fibers"]
        if "n_sky_fibers" in spec:
            hdu_list[0].header["NSKY"] = spec["n_sky_fibers"]

        hdu_list.writeto(fits_path, overwrite=True)

        # ASCII (sky-subtracted only)
        ascii_path = output_dir / f"{base_name}_{band}_coadd.txt"
        table = Table([spec["wave"], spec["flux"], spec["error"]],
                      names=["wavelength", "flux", "error"])
        table.meta["comments"] = [
            "Sky-subtracted coadded spectrum",
            f"MJD-OBS: {exposure.get('mjd_obs', np.nan)}",
            f"EXPTIME: {exposure.get('exptime', np.nan)} s",
        ]
        table.write(ascii_path, format="ascii.fixed_width", overwrite=True)

    logger.info(f"Saved coadds for {base_name} to {output_dir}")


def find_llamas_exposures(extraction_dir, target_object):
    """Scan extraction directory and find complete (blue,green,red) exposure sets."""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning {extraction_dir} for {target_object} exposures...")

    all_files = [f for f in os.listdir(extraction_dir) if f.endswith(".fits")]
    exposure_dict = {}

    for fname in all_files:
        fpath = os.path.join(extraction_dir, fname)
        try:
            hdul = fits.open(fpath)
            obj = hdul[0].header.get("OBJECT", "N/A")
            mjd = hdul[0].header.get("MJD-OBS", None)
            exptime = hdul[0].header.get("REXPTIME", None)
            hdul.close()

            if obj != target_object:
                continue

            parts = fname.split("_")
            if len(parts) < 6:
                logger.warning(f"Skipping {fname}: unexpected filename format")
                continue

            band = parts[-1].replace(".fits", "")  # blue/green/red
            base_name = "_".join(parts[:-2])       # remove 'RSS' and band

            if base_name not in exposure_dict:
                exposure_dict[base_name] = dict(
                    base_name=base_name, mjd_obs=mjd, exptime=exptime,
                    blue=None, green=None, red=None
                )

            exposure_dict[base_name][band] = fpath

        except Exception as e:
            logger.warning(f"Error reading {fname}: {e}")

    complete = []
    for base_name, exp in exposure_dict.items():
        if exp["blue"] and exp["green"] and exp["red"]:
            complete.append(exp)
            logger.info(f"  Found: {base_name} (MJD={exp['mjd_obs']})")
        else:
            missing = [b for b in ["blue", "green", "red"] if not exp[b]]
            logger.warning(f"  Incomplete: {base_name} (missing {', '.join(missing)})")

    logger.info(f"Found {len(complete)} complete exposure sets")
    return complete
