import numpy as np
import matplotlib.pyplot as plt

from utils.whitelight import interpolate_whitelight_to_grid


def plot_whitelight_extraction_diagnostic(
    whitelight,
    selection,
    x0,
    y0,
    bands=("blue", "green", "red"),
    nx=400,
    ny=400,
    smooth_sigma=5.0,
    zoom=5.0,
    fig=None,
    axs=None,
    cmap="magma",
    show_legend=True,
):
    """
    Diagnostic plot for LLAMAS extraction geometry.

    Creates a 2xN panel figure:
      - Row 0: zoomed whitelight patch + fiber scatter + obj/sky overlays
      - Row 1: full whitelight (interpolated, full extent) + overlays

    Parameters
    ----------
    whitelight : dict
        Output from build_whitelight_dict/normalize_whitelight_per_bench:
        whitelight[band] has keys ['xpos','ypos','counts','bench','fiber', ...]
    selection : dict
        Per-band fiber selections, e.g.
        selection[band] = {'obj_inds': array, 'sky_inds': array, 'bench0': str, ...}
    x0, y0 : float
        Spatial center to mark (e.g., fitted Gaussian center).
    bands : tuple/list
        Bands to plot (default: ('blue','green','red')).
    nx, ny : int
        Grid resolution for interpolation.
    smooth_sigma : float
        Gaussian smoothing sigma (pixels) used after nearest-neighbor interpolation.
    zoom : float
        Half-width (in position units) for the patch view limits.
    fig, axs : optional
        Provide an existing figure and axes array. If None, a new one is created.
        If provided, axs must be shape (2, len(bands)).
    cmap : str
        Colormap for interpolated intensity image.
    show_legend : bool
        Whether to show a legend (only on bottom-left panel).

    Returns
    -------
    fig, axs
        Matplotlib figure and axes array of shape (2, len(bands)).
    """
    bands = list(bands)
    n = len(bands)

    # ---- create figure/axes if needed ----
    if fig is None or axs is None:
        fig, axs = plt.subplots(
            nrows=2,
            ncols=n,
            figsize=(5.3 * n, 7.2),
            constrained_layout=True,
        )
    else:
        axs = np.asarray(axs)
        if axs.shape != (2, n):
            raise ValueError(f"axs must have shape (2, {n}), got {axs.shape}")

    # helper: make intensity image from RGBA whitelight grid
    def _grid_intensity(band):
        img, extent, Xg, Yg = interpolate_whitelight_to_grid(
            whitelight, band, nx=nx, ny=ny, sigma=smooth_sigma
        )
        I = img[..., :3].sum(axis=-1)
        I = I - np.nanmedian(I)
        return I, extent

    for c, band in enumerate(bands):
        if band not in whitelight:
            raise KeyError(f"Band '{band}' not present in whitelight dict.")
        if band not in selection:
            raise KeyError(f"Band '{band}' not present in selection dict.")

        obj_inds = np.asarray(selection[band]["obj_inds"], dtype=int)
        sky_inds = np.asarray(selection[band]["sky_inds"], dtype=int)
        bench0 = selection[band].get("bench0", None)

        xpos = whitelight[band]["xpos"]
        ypos = whitelight[band]["ypos"]

        I, extent = _grid_intensity(band)

        # ============================================================
        # Row 0: PATCH (zoomed) view
        # ============================================================
        ax0 = axs[0, c]
        ax0.imshow(
            I,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cmap,
        )

        # all fibers (light)
        ax0.scatter(
            xpos, ypos,
            facecolors="none",
            edgecolors="w",
            s=18,
            alpha=0.35,
            linewidths=0.6,
        )

        # center
        ax0.scatter(
            x0, y0,
            marker="+",
            s=120,
            c="cyan",
            linewidths=2,
        )

        # obj/sky overlays
        ax0.scatter(
            xpos[obj_inds], ypos[obj_inds],
            marker="x",
            s=45,
            c="lime",
            linewidths=2,
        )
        ax0.scatter(
            xpos[sky_inds], ypos[sky_inds],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.5,
            alpha=0.9,
        )

        ax0.set_xlim(x0 - zoom, x0 + zoom)
        ax0.set_ylim(y0 - zoom, y0 + zoom)
        ax0.set_aspect("equal", adjustable="box")
        if bench0 is not None:
            ax0.set_title(f"{band.capitalize()} — patch (bench {bench0})")
        else:
            ax0.set_title(f"{band.capitalize()} — patch")

        # ============================================================
        # Row 1: FULL extent view
        # ============================================================
        ax1 = axs[1, c]
        ax1.imshow(
            I,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=cmap,
        )

        ax1.scatter(
            xpos, ypos,
            facecolors="none",
            edgecolors="w",
            s=15,
            alpha=0.25,
            linewidths=0.6,
            label="fibers" if (c == 0 and show_legend) else None,
        )
        ax1.scatter(
            x0, y0,
            marker="+",
            s=120,
            c="cyan",
            linewidths=2,
            label="center" if (c == 0 and show_legend) else None,
        )
        ax1.scatter(
            xpos[obj_inds], ypos[obj_inds],
            marker="x",
            s=45,
            c="lime",
            linewidths=2,
            label="obj" if (c == 0 and show_legend) else None,
        )
        ax1.scatter(
            xpos[sky_inds], ypos[sky_inds],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.5,
            alpha=0.9,
            label="sky" if (c == 0 and show_legend) else None,
        )

        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(f"{band.capitalize()} — full whitelight")

        if c == 0 and show_legend:
            ax1.legend(loc="best", fontsize=9)

    # optional: tidy axis labels (usually not needed for these diagnostic images)
    for a in axs.ravel():
        a.set_xlabel("")
        a.set_ylabel("")

    return fig, axs
    
def plot_obj_sky_and_subtracted(coadds, title_suffix="", save_path=None):
    fig, ax = plt.subplots(nrows=3, figsize=(10, 8), sharex=False)
    bands = ['blue', 'green', 'red']
    for r, band in enumerate(bands):
        wave = coadds[band]["wave"]
        obj  = coadds[band]["obj_flux"]
        sky  = coadds[band]["sky_flux"]

        ax[r].plot(wave, obj, lw=1.4, label="obj")
        ax[r].plot(wave, sky, lw=1.0, ls="--", alpha=0.8, label="sky")
        ax[r].set_ylabel("Flux")
        ax[r].set_title(f"{band}{title_suffix}")
        ax[r].legend()

        if band in ["blue", "green"]:
            ax[r].axvline(4686, ls="--", alpha=0.4)

    ax[-1].set_xlabel("Wavelength (Å)")
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

    fig, ax = plt.subplots(nrows=3, figsize=(10, 8), sharex=False)
    for r, band in enumerate(bands):
        wave = coadds[band]["wave"]
        sub  = coadds[band]["obj_flux"] - coadds[band]["sky_flux"]
        ax[r].plot(wave, sub, lw=1.4, label="obj-sky")
        ax[r].set_ylabel("Flux")
        ax[r].set_title(f"{band} sky-subtracted{title_suffix}")
        ax[r].legend()
        if band in ["blue", "green"]:
            ax[r].axvline(4686, ls="--", alpha=0.4)

    ax[-1].set_xlabel("Wavelength (Å)")
    plt.tight_layout()
    
    if save_path:
        # Save subtracted plot with _subtracted suffix
        save_path_sub = save_path.parent / f"{save_path.stem}_subtracted{save_path.suffix}"
        plt.savefig(save_path_sub, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path_sub}")
    
    plt.show()


def plot_coadd_diagnostic(
    spec_map,
    coadds,
    selection,
    bands=("blue", "green", "red"),
    show_residuals=True,
    vmin_percentile=1,
    vmax_percentile=99,
    figsize=None,
):
    """
    Diagnostic plot for LLAMAS fiber coadding.

    Creates a multi-panel figure showing:
      - 2D spectral stack (flux vs wavelength for each fiber)
      - Weight distribution for each fiber
      - Residuals from the coadd (if requested)

    Parameters
    ----------
    spec_map : dict
        Raw spectral data per band, e.g. spec_map[band] = {'flux': array, 'wave': array}
    coadds : dict
        Coadd results per band from coadd_obj_sky, must include:
        coadds[band] = {'wave', 'obj_flux', 'sky_flux', 'obj_weights', 'sky_weights', ...}
    selection : dict
        Fiber selection per band, e.g. selection[band] = {'obj_inds': array, 'sky_inds': array}
    bands : tuple
        Bands to plot
    show_residuals : bool
        Whether to show residual panels (individual fiber - coadd)
    vmin_percentile, vmax_percentile : float
        Percentile clipping for 2D image color scaling
    figsize : tuple or None
        Figure size override

    Returns
    -------
    fig, axs
        Matplotlib figure and axes
    """
    bands = list(bands)
    n_bands = len(bands)

    # Determine layout: 4 rows per band (obj stack, obj weights, sky stack, sky weights)
    # + optional residual rows
    n_rows_per_band = 4 if not show_residuals else 6
    n_rows_total = n_rows_per_band * n_bands

    if figsize is None:
        figsize = (14, 3 * n_rows_total)

    fig, axs = plt.subplots(
        nrows=n_rows_total,
        ncols=1,
        figsize=figsize,
        constrained_layout=True,
    )
    axs = np.atleast_1d(axs)

    ax_idx = 0

    for band in bands:
        if band not in spec_map:
            raise KeyError(f"Band '{band}' not in spec_map")
        if band not in coadds:
            raise KeyError(f"Band '{band}' not in coadds")
        if band not in selection:
            raise KeyError(f"Band '{band}' not in selection")

        flux_arr = spec_map[band]["flux"]
        wave_arr = spec_map[band]["wave"]
        obj_inds = selection[band]["obj_inds"]
        sky_inds = selection[band]["sky_inds"]

        wave_ref = coadds[band]["wave"]
        obj_flux_coadd = coadds[band]["obj_flux"]
        sky_flux_coadd = coadds[band]["sky_flux"]

        # Get weights (handle if not present for backward compatibility)
        obj_weights = coadds[band].get("obj_weights", None)
        sky_weights = coadds[band].get("sky_weights", None)

        # Interpolate individual fiber spectra onto reference wavelength
        def _interp_fibers(inds):
            n_fib = len(inds)
            flux_grid = np.empty((n_fib, len(wave_ref)), dtype=float)
            for i, idx in enumerate(inds):
                w_i, f_i = wave_arr[idx], flux_arr[idx]
                if w_i[0] > w_i[-1]:
                    w_i, f_i = w_i[::-1], f_i[::-1]
                flux_grid[i] = np.interp(wave_ref, w_i, f_i)
            return flux_grid

        obj_flux_grid = _interp_fibers(obj_inds)
        sky_flux_grid = _interp_fibers(sky_inds)

        # ========== OBJECT STACK ==========
        ax = axs[ax_idx]
        vmin_obj = np.nanpercentile(obj_flux_grid, vmin_percentile)
        vmax_obj = np.nanpercentile(obj_flux_grid, vmax_percentile)
        im = ax.imshow(
            obj_flux_grid,
            aspect="auto",
            origin="lower",
            extent=[wave_ref[0], wave_ref[-1], 0, len(obj_inds)],
            vmin=vmin_obj,
            vmax=vmax_obj,
            cmap="viridis",
        )
        ax.set_ylabel(f"Obj fiber index")
        ax.set_title(f"{band.capitalize()} — Object fibers (2D stack)")
        plt.colorbar(im, ax=ax, label="Flux")
        ax_idx += 1

        # ========== OBJECT WEIGHTS ==========
        ax = axs[ax_idx]
        if obj_weights is not None:
            # weights is 2D (n_fiber, n_wave), compute total weight per fiber
            total_weights = np.nansum(obj_weights, axis=1)
            ax.bar(range(len(obj_inds)), total_weights, color="steelblue", edgecolor="k")
            ax.set_ylabel("Total weight")
            ax.set_xlabel("Obj fiber index")
            ax.set_title(f"{band.capitalize()} — Object fiber weights (summed over wavelength)")
        else:
            ax.text(
                0.5, 0.5, "Weights not available\n(update coadd_obj_sky)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        ax_idx += 1

        # ========== SKY STACK ==========
        ax = axs[ax_idx]
        vmin_sky = np.nanpercentile(sky_flux_grid, vmin_percentile)
        vmax_sky = np.nanpercentile(sky_flux_grid, vmax_percentile)
        im = ax.imshow(
            sky_flux_grid,
            aspect="auto",
            origin="lower",
            extent=[wave_ref[0], wave_ref[-1], 0, len(sky_inds)],
            vmin=vmin_sky,
            vmax=vmax_sky,
            cmap="viridis",
        )
        ax.set_ylabel(f"Sky fiber index")
        ax.set_title(f"{band.capitalize()} — Sky fibers (2D stack)")
        plt.colorbar(im, ax=ax, label="Flux")
        ax_idx += 1

        # ========== SKY WEIGHTS ==========
        ax = axs[ax_idx]
        if sky_weights is not None:
            # For median coadd, weights might be uniform or None
            # Just show as placeholder or indicator
            ax.text(
                0.5, 0.5, f"Sky: median coadd\n({len(sky_inds)} fibers)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        else:
            ax.text(
                0.5, 0.5, f"Sky: median coadd\n({len(sky_inds)} fibers)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        ax_idx += 1

        # ========== RESIDUALS (optional) ==========
        if show_residuals:
            # Object residuals
            ax = axs[ax_idx]
            obj_resid = obj_flux_grid - obj_flux_coadd[None, :]
            vmin_r = np.nanpercentile(obj_resid, 5)
            vmax_r = np.nanpercentile(obj_resid, 95)
            im = ax.imshow(
                obj_resid,
                aspect="auto",
                origin="lower",
                extent=[wave_ref[0], wave_ref[-1], 0, len(obj_inds)],
                vmin=vmin_r,
                vmax=vmax_r,
                cmap="RdBu_r",
            )
            ax.set_ylabel(f"Obj fiber index")
            ax.set_title(f"{band.capitalize()} — Object residuals (fiber - coadd)")
            plt.colorbar(im, ax=ax, label="Residual flux")
            ax_idx += 1

            # Sky residuals
            ax = axs[ax_idx]
            sky_resid = sky_flux_grid - sky_flux_coadd[None, :]
            vmin_r = np.nanpercentile(sky_resid, 5)
            vmax_r = np.nanpercentile(sky_resid, 95)
            im = ax.imshow(
                sky_resid,
                aspect="auto",
                origin="lower",
                extent=[wave_ref[0], wave_ref[-1], 0, len(sky_inds)],
                vmin=vmin_r,
                vmax=vmax_r,
                cmap="RdBu_r",
            )
            ax.set_ylabel(f"Sky fiber index")
            ax.set_xlabel(f"Wavelength (Å)")
            ax.set_title(f"{band.capitalize()} — Sky residuals (fiber - coadd)")
            plt.colorbar(im, ax=ax, label="Residual flux")
            ax_idx += 1

    return fig, axs