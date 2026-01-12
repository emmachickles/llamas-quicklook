import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .io import load_llamas_spectrum
from .geometry import get_fiber_positions


def build_whitelight_dict(f_specb, f_specg, f_specr, fibermap_lut):
    """Construct whitelight (counts per fiber + positions) for each band."""
    bands = ["blue", "green", "red"]
    filepaths = [f_specb, f_specg, f_specr]
    whitelight = {}

    for band, filepath in zip(bands, filepaths):
        spec_data = load_llamas_spectrum(filepath)
        flux = spec_data["flux"]
        fibermap = spec_data["fibermap"]

        xpos, ypos, fiber, bench = get_fiber_positions(fibermap, fibermap_lut)

        counts = np.nansum(flux, axis=1)

        whitelight[band] = dict(
            xpos=xpos, ypos=ypos, counts=counts, fiber=fiber, bench=bench
        )

    return whitelight


def normalize_whitelight_per_bench(whitelight_dict, cmap_name="magma"):
    """Normalize counts per bench and attach RGBA colors."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)

    for band in list(whitelight_dict.keys()):
        cnts_arr = whitelight_dict[band]["counts"]
        bench_arr = whitelight_dict[band]["bench"]
        benches = np.unique(bench_arr)

        bench_meds, bench_spreads = [], []
        for b in benches:
            inds = np.nonzero(bench_arr == b)[0]
            cnts = cnts_arr[inds]
            med = np.nanmedian(cnts)
            mad = np.nanmedian(np.abs(cnts - med))
            bench_meds.append(med)
            bench_spreads.append(mad)

        bench_meds = np.array(bench_meds)
        bench_spreads = np.array(bench_spreads)

        global_ref = np.nanmedian(bench_meds)
        global_spread = np.nanmedian(bench_spreads)

        cnts_scaled = np.zeros_like(cnts_arr)
        for b, med, spread in zip(benches, bench_meds, bench_spreads):
            inds = np.nonzero(bench_arr == b)[0]
            cnts = cnts_arr[inds]
            scale = global_spread / spread if spread > 0 else 1.0
            cnts_scaled[inds] = global_ref + (cnts - med) * scale

        vmin, vmax = np.percentile(cnts_scaled, [0.01, 99.7])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        whitelight_dict[band]["colors"] = cmap(norm(cnts_scaled))

    return whitelight_dict


def interpolate_whitelight_to_grid(whitelight_dict, band, nx=400, ny=400, sigma=5.0):
    """
    Interpolate sparse fiber whitelight RGBA to a regular 2D grid.
    Returns (img_rgba, extent, Xg, Yg).
    """
    xpos_arr = whitelight_dict[band]["xpos"]
    ypos_arr = whitelight_dict[band]["ypos"]
    color_arr = whitelight_dict[band]["colors"]

    xmin, xmax = xpos_arr.min(), xpos_arr.max()
    ymin, ymax = ypos_arr.min(), ypos_arr.max()
    pad = 0.02 * max(xmax - xmin, ymax - ymin)

    xi = np.linspace(xmin - pad, xmax + pad, nx)
    yi = np.linspace(ymin - pad, ymax + pad, ny)
    Xg, Yg = np.meshgrid(xi, yi)

    points = np.column_stack((xpos_arr, ypos_arr))
    img = np.empty((ny, nx, 4), dtype=float)

    for ch in range(4):
        Ci = griddata(points, color_arr[:, ch], (Xg, Yg), method="nearest")
        img[..., ch] = gaussian_filter(Ci, sigma=sigma)

    extent = (xi[0], xi[-1], yi[0], yi[-1])
    return img, extent, Xg, Yg
