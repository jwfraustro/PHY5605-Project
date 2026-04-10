"""Utility functions for the program."""

import astropy.io.fits as fits
import numpy as np
from scipy.interpolate import splev, splrep


def save_fits(filename: str, data: np.ndarray):
    """A helper function for saving FITS files."""
    fits.writeto(filename, np.float32(data), overwrite=True, output_verify="silentfix")


def cross_corr(
    f: np.ndarray,
    g: np.ndarray,
    shifts: np.ndarray | None = None,
) -> np.ndarray:
    """Cross correlation function.

    Accepts:
    f, g: array_like
    shifts: array_like

    Returns:
    c: array_like, correlation values."""
    if shifts is None:
        shifts = np.arange(len(g))
    c = np.zeros(len(shifts), dtype=float)
    f_pad = np.zeros(3 * len(f))
    g_pad = np.zeros(3 * len(f))
    f_pad[len(f) : 2 * len(f)] = f - f.mean()
    g_pad[len(g) : 2 * len(g)] = g - g.mean()
    for i in range(len(shifts)):
        c[i] = np.correlate(f_pad, np.roll(g_pad, -shifts[i]))

    return c


def wavelength_shift(
    in_pixel: float, ref_pixel: float, ref_wavelength: float, dispersion: float
) -> float:
    """Determine wavelength shift from a pixel offset.

    Parameters
    ----------
    in_pixel : float
        Pixel position to convert.
    ref_pixel : float
        Reference pixel position.
    ref_wavelength : float
        Wavelength at the reference pixel.
    dispersion : float
        Dispersion in wavelength units per pixel.

    Returns
    -------
    float
        Wavelength offset from the reference pixel.
    """
    pixel_diff = in_pixel - ref_pixel
    wavelength = pixel_diff * dispersion

    return wavelength


def norm_med_comb(
    data: np.ndarray,
    region: tuple[tuple[int | None, int | None], tuple[int | None, int | None]] = (
        (None, None),
        (None, None),
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized median combination of 3-D data.

    Each frame is divided by the median of the specified region,
    then the stack is median-combined along the first axis.

    Parameters
    ----------
    data : np.ndarray
        3-D array with shape ``(n_frames, ny, nx)``.
    region : tuple of two (int | None, int | None) tuples, optional
        ``((y1, x1), (y2, x2))`` slice bounds for the normalization
        region.  ``None`` values default to the array edges.

    Returns
    -------
    med_comb_data : np.ndarray
        2-D median-combined image.
    norm_factors : np.ndarray
        1-D array of per-frame normalization factors.
    """
    ((y1, x1), (y2, x2)) = region

    num_frames = data.shape[0]
    norm_factors = np.zeros(num_frames)
    norm_data = data.copy()

    for frame in range(num_frames):
        norm_factors[frame] = np.median(norm_data[frame, y1:y2, x1:x2])
        norm_data[frame] /= norm_factors[frame]

    med_comb_data = np.median(norm_data, axis=0)

    return med_comb_data, norm_factors


def splinterp(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """Interpolate data using a cubic spline.

    Parameters
    ----------
    x_new : np.ndarray
        Positions at which to evaluate the spline.
    x_old : np.ndarray
        Known sample positions.
    y_old : np.ndarray
        Known sample values.

    Returns
    -------
    np.ndarray
        Interpolated values at *x_new*.
    """
    spline = splrep(x_old, y_old)
    y_new = splev(x_new, spline)

    return y_new
