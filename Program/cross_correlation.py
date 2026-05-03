"""Cross-correlation with sub-pixel precision via Gaussian peak fitting.

This module replaces the original cross_corr() from project_funcs.py,
which returned raw correlation values and relied on argmax for peak
finding — destroying all sub-pixel information.

The new implementation follows the assignment spec (Problem 3):
  1. Interpolate both spectra onto a 10x finer grid using splinterp
  2. Subtract the mean from each interpolated spectrum
  3. Zero-pad to prevent wrapping
  4. Compute the cross-correlation over a range of lags
  5. Fit a Gaussian to the peak region for sub-pixel precision
  6. Return the shift in original (not interpolated) pixel units
"""

import warnings

import numpy as np
from gaussian import fitgaussian
from project_funcs import splinterp


def _raw_cross_correlate(
    f: np.ndarray,
    g: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """Compute the cross-correlation between f and g at given shifts.

    Both f and g should already be mean-subtracted and zero-padded.

    Parameters
    ----------
    f : np.ndarray
        Reference spectrum (mean-subtracted, zero-padded).
    g : np.ndarray
        Comparison spectrum (mean-subtracted, zero-padded).
    shifts : np.ndarray
        Array of integer lag values to evaluate.

    Returns
    -------
    c : np.ndarray
        Cross-correlation values at each shift.
    """
    c = np.empty(len(shifts), dtype=float)
    for i, s in enumerate(shifts):
        c[i] = np.correlate(f, np.roll(g, -int(s)))[0]
    return c


def cross_correlate_subpixel(
    spectrum: np.ndarray,
    reference: np.ndarray,
    shift_range: tuple[int, int] = (-10, 10),
    supersample: int = 10,
    gauss_fit_halfwidth: int = 5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Find the sub-pixel shift between a spectrum and a reference.

    Interpolates both spectra to a finer grid, cross-correlates over
    the specified shift range, and fits a Gaussian to the peak of the
    correlation function for sub-pixel precision.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D spectrum to measure the shift of.
    reference : np.ndarray
        1-D reference spectrum (shift is measured relative to this).
    shift_range : tuple[int, int]
        (min_shift, max_shift) in original pixel units. The range of
        lags to consider. Should be substantially larger than the
        expected shift.
    supersample : int
        Interpolation factor. 10 means 1/10-pixel resolution on the
        fine grid.
    gauss_fit_halfwidth : int
        Number of fine-grid points on each side of the correlation
        peak to include in the Gaussian fit. Larger values are more
        robust to noise but may fail if the peak is narrow.

    Returns
    -------
    shift : float
        Sub-pixel shift in original pixel units. Positive means
        `spectrum` is shifted to the right relative to `reference`.
    shift_err : float
        Uncertainty on the shift from the Gaussian fit, in original
        pixel units. Set to NaN if the fit failed.
    shifts_fine : np.ndarray
        The fine-grid shift values (for diagnostic plotting).
    cc_values : np.ndarray
        The cross-correlation values at each fine-grid shift.
    """

    n_pix = len(spectrum)

    # Step 1: Interpolate both spectra onto the fine grid
    x_orig = np.arange(n_pix, dtype=float)
    x_fine = np.linspace(0, n_pix - 1, n_pix * supersample)

    spec_fine = splinterp(x_fine, x_orig, spectrum)
    ref_fine = splinterp(x_fine, x_orig, reference)

    # Step 2: Subtract means
    spec_fine = spec_fine - spec_fine.mean()
    ref_fine = ref_fine - ref_fine.mean()

    # Step 3: Zero-pad to prevent wrapping
    n_fine = len(spec_fine)
    f_pad = np.zeros(3 * n_fine)
    g_pad = np.zeros(3 * n_fine)
    f_pad[n_fine : 2 * n_fine] = spec_fine
    g_pad[n_fine : 2 * n_fine] = ref_fine

    # Step 4: Cross-correlate over the shift range (in fine-grid units)
    shift_min_fine = shift_range[0] * supersample
    shift_max_fine = shift_range[1] * supersample
    shifts_fine = np.arange(shift_min_fine, shift_max_fine + 1)

    cc_values = _raw_cross_correlate(f_pad, g_pad, shifts_fine)

    # Step 5: Fit Gaussian to the peak region
    peak_idx = np.argmax(cc_values)
    shift, shift_err = _fit_peak(
        cc_values, shifts_fine, peak_idx, gauss_fit_halfwidth, supersample
    )

    return shift, shift_err, shifts_fine, cc_values


def _fit_peak(
    cc_values: np.ndarray,
    shifts_fine: np.ndarray,
    peak_idx: int,
    halfwidth: int,
    supersample: int,
) -> tuple[float, float]:
    """Fit a Gaussian to the cross-correlation peak.

    Tries progressively smaller fitting windows if the fit fails,
    as recommended by the assignment spec. Falls back to the raw
    argmax if all fits fail.

    Parameters
    ----------
    cc_values : np.ndarray
        Cross-correlation values.
    shifts_fine : np.ndarray
        Corresponding shift values on the fine grid.
    peak_idx : int
        Index of the peak in cc_values.
    halfwidth : int
        Initial number of points on each side of the peak to fit.
    supersample : int
        Interpolation factor, to convert fine-grid shifts back to
        original pixel units.

    Returns
    -------
    shift : float
        Sub-pixel shift in original pixel units.
    shift_err : float
        Uncertainty on the shift, or NaN if fit failed.
    """

    # Try progressively smaller windows
    for hw in [halfwidth, max(halfwidth // 2, 3), 3]:
        lo = max(peak_idx - hw, 0)
        hi = min(peak_idx + hw + 1, len(cc_values))

        cc_region = cc_values[lo:hi].copy()
        shift_region = shifts_fine[lo:hi].astype(float)

        if len(cc_region) < 3:
            continue

        try:
            width, center, height, err = fitgaussian(cc_region, shift_region)
            fitted_center = center[0]
            center_err = err[1]  # err is [width_err, center_err, height_err]

            # Sanity check: fitted center should be within the fitting window
            if fitted_center < shift_region[0] or fitted_center > shift_region[-1]:
                continue

            # Negate: the correlator finds the lag where g aligns with f,
            # but we want "how is spectrum shifted relative to reference."
            # Rolling g by -s and peaking at +s means g (reference) must
            # move right to match f (spectrum), so spectrum is shifted left
            # — hence negate to get spectrum's shift.
            return -fitted_center / supersample, center_err / supersample

        except (ValueError, RuntimeError):
            continue

    # All fits failed — fall back to argmax (integer precision)
    warnings.warn(
        f"Gaussian fit failed at all window sizes; "
        f"falling back to argmax (shift={-shifts_fine[peak_idx] / supersample:.1f})",
        stacklevel=3,
    )
    return -shifts_fine[peak_idx] / supersample, np.nan