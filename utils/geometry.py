import numpy as np

def get_fibermap_lut():
    from astropy.table import Table
    from utils.paths import config_path
    import os
    lut_path = config_path("LLAMAS_FiberMap_rev04.dat")
    return Table.read(lut_path, format="ascii.fixed_width")

def get_fiber_positions(fibermap_data, fibermap_lut):
    """
    Map fiber IDs and bench sides to physical positions using LUT.
    Returns xpos, ypos, fiber_id, benchside arrays.
    """
    xpos_arr, ypos_arr, fiber_arr, bench_arr = [], [], [], []

    for i in range(len(fibermap_data)):
        fiber_id = fibermap_data[i][0]
        benchside = fibermap_data[i][1]

        match = np.where(
            (fibermap_lut["bench"] == benchside) &
            (fibermap_lut["fiber"] == fiber_id)
        )[0]

        if len(match) > 0:
            ind = match[0]
            xpos_arr.append(fibermap_lut["xpos"][ind])
            ypos_arr.append(fibermap_lut["ypos"][ind])
            fiber_arr.append(fiber_id)
            bench_arr.append(benchside)

    return (np.array(xpos_arr), np.array(ypos_arr),
            np.array(fiber_arr), np.array(bench_arr))

def _gaussian2d(coords, amp, x0, y0, sx, sy, offset):
    x, y = coords
    return amp * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))) + offset


def fit_whitelight_center_2dgauss(
    whitelight,
    band,
    x_guess,
    y_guess,
    half_size=3.0,
    nx=400,
    ny=400,
    smooth_sigma=5.0,
    sx0=1.0,
    sy0=1.0,
    clip_negative=True,
    clip_high_quantile=None,  # e.g. 0.995 to reduce single-pixel domination
    make_plot=False,
    ax=None,
):
    """
    Fit a 2D Gaussian to a whitelight intensity patch around (x_guess, y_guess)
    using your interpolate_whitelight_to_grid() RGBA output.

    Parameters
    ----------
    whitelight : dict
        Whitelight dict passed to interpolate_whitelight_to_grid.
    band : str
        'blue'/'green'/'red'
    x_guess, y_guess : float
        Initial guess in the same coordinate system as xpos/ypos.
    half_size : float
        Half-width of the fit patch in DATA units (same units as Xg/Yg).
    nx, ny : int
        Grid size used in interpolation.
    smooth_sigma : float
        Gaussian smoothing sigma (grid pixels) inside interpolate_whitelight_to_grid.
    sx0, sy0 : float
        Initial sigma guesses for the 2D Gaussian in DATA units.
    clip_negative : bool
        If True, subtract patch median and clip to >= 0 before fitting (stabilizes).
    clip_high_quantile : float or None
        If set (0<q<1), clip patch values above this quantile before fitting.
    make_plot : bool
        If True, make a quicklook plot zoomed to the patch.
    ax : matplotlib Axes or None
        If provided, plot into this axes; otherwise create a new fig/ax.

    Returns
    -------
    result : dict
        Keys:
          - x_fit, y_fit, sx_fit, sy_fit, amp, offset
          - popt, pcov
          - I (full intensity image), extent, Xg, Yg
          - patch_slices: (slice_y, slice_x)
          - I_patch, X_patch, Y_patch
    """

    from .whitelight import interpolate_whitelight_to_grid
    from scipy.optimize import curve_fit
    
    # ---- interpolate + intensity image
    img_rgba, extent, Xg, Yg = interpolate_whitelight_to_grid(
        whitelight, band, nx=nx, ny=ny, sigma=smooth_sigma
    )

    I = img_rgba[..., :3].sum(axis=-1).astype(float)
    I = I - np.nanmedian(I)

    # ---- define patch in DATA space, then convert to index bbox
    patch_mask = (
        (Xg >= x_guess - half_size) & (Xg <= x_guess + half_size) &
        (Yg >= y_guess - half_size) & (Yg <= y_guess + half_size)
    )

    ys, xs = np.where(patch_mask)
    if len(xs) == 0:
        raise ValueError(
            "Patch is empty. Increase half_size or check x_guess/y_guess against extent."
        )

    y0i, y1i = ys.min(), ys.max() + 1
    x0i, x1i = xs.min(), xs.max() + 1
    sy_slice = slice(y0i, y1i)
    sx_slice = slice(x0i, x1i)

    I_patch = I[sy_slice, sx_slice]
    X_patch = Xg[sy_slice, sx_slice]
    Y_patch = Yg[sy_slice, sx_slice]

    # ---- stabilize patch for fitting
    fit_patch = I_patch.copy()

    # offset estimate for initial guess (and optional preprocessing)
    offset0 = np.nanmedian(fit_patch)

    if clip_negative:
        fit_patch = fit_patch - offset0
        fit_patch = np.clip(fit_patch, 0, None)
        # after clipping, keep offset parameter near 0 for the model baseline
        offset0_fit = 0.0
    else:
        offset0_fit = offset0

    if clip_high_quantile is not None:
        good = np.isfinite(fit_patch)
        if np.any(good):
            hi = np.nanquantile(fit_patch[good], clip_high_quantile)
            fit_patch = np.clip(fit_patch, None, hi)

    # ---- initial parameter guess
    amp0 = np.nanmax(fit_patch) - np.nanmedian(fit_patch)
    if not np.isfinite(amp0) or amp0 <= 0:
        # fallback: use raw patch
        amp0 = np.nanmax(I_patch) - np.nanmedian(I_patch)

    jmax = np.nanargmax(fit_patch)
    ymax, xmax = np.unravel_index(jmax, fit_patch.shape)
    x0_init = X_patch[ymax, xmax]
    y0_init = Y_patch[ymax, xmax]

    p0 = (amp0, x0_init, y0_init, sx0, sy0, offset0_fit)

    # ---- bounds
    dx = np.nanmedian(np.diff(Xg[0, :]))
    dy = np.nanmedian(np.diff(Yg[:, 0]))
    dx = float(dx) if np.isfinite(dx) and dx != 0 else 1.0
    dy = float(dy) if np.isfinite(dy) and dy != 0 else 1.0

    lower = (0.0, x_guess - half_size, y_guess - half_size, dx / 5, dy / 5, -np.inf)
    upper = (np.inf, x_guess + half_size, y_guess + half_size, half_size * 2, half_size * 2, np.inf)

    # ---- fit
    popt, pcov = curve_fit(
        _gaussian2d,
        (X_patch.ravel(), Y_patch.ravel()),
        fit_patch.ravel(),
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )

    amp, x_fit, y_fit, sx_fit, sy_fit, offset_fit = popt

    # If we subtracted median & clipped negatives, offset_fit is relative to that.
    # For most uses (centroiding), you only care about x_fit/y_fit anyway.

    result = dict(
        x_fit=x_fit,
        y_fit=y_fit,
        sx_fit=sx_fit,
        sy_fit=sy_fit,
        amp=amp,
        offset=offset_fit,
        popt=popt,
        pcov=pcov,
        I=I,
        extent=extent,
        Xg=Xg,
        Yg=Yg,
        patch_slices=(sy_slice, sx_slice),
        I_patch=I_patch,
        X_patch=X_patch,
        Y_patch=Y_patch,
    )

    # ---- optional plot (zoomed to patch)
    if make_plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(I, extent=extent, origin="lower", aspect="equal")
        ax.scatter([x_guess], [y_guess], marker="+", s=80, label="guess")
        ax.scatter([x_fit], [y_fit], marker="x", s=80, label="2D Gauss fit")

        # zoom to patch bounds
        ax.set_xlim(Xg[y0i, x0i], Xg[y0i, x1i - 1])
        ax.set_ylim(Yg[y0i, x0i], Yg[y1i - 1, x0i])
        ax.legend()

    return result

def select_fiber_aperture_and_sky_annulus(
    xpos,
    ypos,
    bench,
    x0,
    y0,
    r_obj_max=3.0,
    r_sky_min=5.0,
    r_sky_max=15.0,
):
    """
    Select object and sky fibers based on distance from (x0, y0).

    Parameters
    ----------
    x0, y0 : float
        Spatial center
    n_obj : int
        Number of object fibers
    Returns
    -------
    obj_inds : ndarray[int]
    sky_inds : ndarray[int]
    bench0 : str
    center_ind : int
    """

    d_all = np.sqrt((xpos - x0)**2 + (ypos - y0)**2)
    center_ind = int(np.argmin(d_all))
    bench0 = bench[center_ind]

    bench_inds = np.where(bench == bench0)[0]
    bx, by = xpos[bench_inds], ypos[bench_inds]
    d_bench = np.sqrt((bx - x0)**2 + (by - y0)**2)

    obj_mask = d_bench <= r_obj_max
    n_obj = np.sum(obj_mask)
    obj_inds = bench_inds[obj_mask]
    
    sky_mask = (d_bench >= r_sky_min) & (d_bench <= r_sky_max) 
    sky_inds = bench_inds[sky_mask]

    return obj_inds, sky_inds, bench0, center_ind
