import numpy as np


def weighted_coadd(flux_arr, indices, wave_arr=None, wave_ref=None,
                   sigma_clip=None, min_weight=0.0):
    """Whitelight-weighted coadd (weights = sum of spectrum per fiber)."""
    sel_flux = flux_arr[indices]

    if wave_arr is not None and wave_ref is not None:
        wave_sel = wave_arr[indices]
        n_sel, n_ref = sel_flux.shape[0], wave_ref.shape[0]
        flux_interp = np.empty((n_sel, n_ref), dtype=float)

        for i in range(n_sel):
            w_i, f_i = wave_sel[i], sel_flux[i]
            if w_i[0] > w_i[-1]:
                w_i, f_i = w_i[::-1], f_i[::-1]
            flux_interp[i] = np.interp(wave_ref, w_i, f_i)

        sel_flux = flux_interp

    wl = np.nansum(sel_flux, axis=1)
    wl = np.clip(wl, min_weight, None)
    if not np.any(wl > 0):
        wl = np.ones_like(wl)

    w = wl[:, None] * np.ones_like(sel_flux)

    if sigma_clip is not None:
        n_fib, n_wave = sel_flux.shape
        clip_mask = np.ones_like(sel_flux, dtype=bool)
        for j in range(n_wave):
            col = sel_flux[:, j]
            med = np.nanmedian(col)
            mad = np.nanmedian(np.abs(col - med))
            if mad > 0:
                clip_mask[:, j] = np.abs(col - med) <= sigma_clip * 1.4826 * mad
        w = np.where(clip_mask, w, 0.0)

    num = np.nansum(w * sel_flux, axis=0)
    den = np.nansum(w, axis=0)
    coadd_flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    resid = sel_flux - coadd_flux[None, :]
    mad_wave = np.nanmedian(np.abs(resid), axis=0)
    robust_sigma = 1.4826 * mad_wave
    n_eff = np.maximum(np.sum(w > 0, axis=0), 1)
    coadd_err = robust_sigma / np.sqrt(n_eff)

    return coadd_flux, coadd_err, w


def ivw_coadd(flux_arr, err_arr, indices, err_floor=0.0,
              wave_arr=None, wave_ref=None, sigma_clip=None):
    """Inverse-variance weighted coadd (optionally interpolated onto wave_ref)."""
    sel_flux = flux_arr[indices]
    sel_err = err_arr[indices]

    if wave_arr is not None and wave_ref is not None:
        wave_sel = wave_arr[indices]
        n_sel, n_ref = sel_flux.shape[0], wave_ref.shape[0]
        flux_interp = np.empty((n_sel, n_ref), dtype=float)
        err_interp = np.empty((n_sel, n_ref), dtype=float)

        for i in range(n_sel):
            w_i, f_i, e_i = wave_sel[i], sel_flux[i], sel_err[i]
            if w_i[0] > w_i[-1]:
                w_i, f_i, e_i = w_i[::-1], f_i[::-1], e_i[::-1]
            flux_interp[i] = np.interp(wave_ref, w_i, f_i)
            err_interp[i] = np.interp(wave_ref, w_i, e_i)

        sel_flux, sel_err = flux_interp, err_interp

    if sigma_clip is not None:
        n_fibers, n_wave = sel_flux.shape
        clip_mask = np.ones_like(sel_flux, dtype=bool)
        for j in range(n_wave):
            flux_at_wave = sel_flux[:, j]
            med = np.nanmedian(flux_at_wave)
            mad = np.nanmedian(np.abs(flux_at_wave - med))
            if mad > 0:
                clip_mask[:, j] = np.abs(flux_at_wave - med) <= sigma_clip * 1.4826 * mad
    else:
        clip_mask = None

    denom = np.square(sel_err)
    if err_floor > 0:
        denom = denom + err_floor**2

    w = np.zeros_like(sel_err, dtype=float)
    mask = np.isfinite(denom) & (denom > 0)
    if clip_mask is not None:
        mask = mask & clip_mask
    w[mask] = 1.0 / denom[mask]

    num = np.nansum(w * sel_flux, axis=0)
    den = np.nansum(w, axis=0)

    coadd_flux = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    coadd_err = np.sqrt(1.0 / np.maximum(den, 1e-300))
    return coadd_flux, coadd_err


def median_coadd(flux_arr, indices, wave_arr=None, wave_ref=None):
    """Median coadd (optionally interpolated onto wave_ref)."""
    sel_flux = flux_arr[indices]

    if wave_arr is not None and wave_ref is not None:
        wave_sel = wave_arr[indices]
        n_sel, n_ref = sel_flux.shape[0], wave_ref.shape[0]
        flux_interp = np.empty((n_sel, n_ref), dtype=float)

        for i in range(n_sel):
            w_i, f_i = wave_sel[i], sel_flux[i]
            if w_i[0] > w_i[-1]:
                w_i, f_i = w_i[::-1], f_i[::-1]
            flux_interp[i] = np.interp(wave_ref, w_i, f_i)

        sel_flux = flux_interp

    coadd_flux = np.nanmedian(sel_flux, axis=0)
    mad = np.nanmedian(np.abs(sel_flux - coadd_flux), axis=0)
    coadd_err = 1.4826 * mad / np.sqrt(sel_flux.shape[0])
    return coadd_flux, coadd_err

def coadd_obj_sky(
    spec_band,
    obj_inds,
    sky_inds,
    sigma_clip=3.0,
):
    """
    Coadd object and sky fibers for one band.

    Returns
    -------
    dict with keys:
        wave, obj_flux, obj_err, sky_flux, sky_err,
        obj_inds, sky_inds
    """
    flux_arr = spec_band["flux"]
    wave_arr = spec_band["wave"]

    wave_ref = wave_arr[obj_inds[0]]

    obj_flux, obj_err, obj_weights = weighted_coadd(
        flux_arr,
        obj_inds,
        wave_arr=wave_arr,
        wave_ref=wave_ref,
        sigma_clip=sigma_clip,
    )

    sky_flux, sky_err = median_coadd(
        flux_arr,
        sky_inds,
        wave_arr=wave_arr,
        wave_ref=wave_ref
    )
                                     

    return {
        "wave": wave_ref,
        "obj_flux": obj_flux,
        "obj_err": obj_err,
        "sky_flux": sky_flux,
        "sky_err": sky_err,
        "obj_inds": np.array(obj_inds),
        "sky_inds": np.array(sky_inds),
        "obj_weights": obj_weights,
        "sky_weights": None,  # median coadd has no weights
    }
