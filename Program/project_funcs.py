import numpy as np
import astropy.io.fits as fits
from scipy.interpolate import splev, splrep

def save_fits(filename, data):
    "A helper function for saving FITS files."
    fits.writeto(filename, np.float32(data), overwrite=True, output_verify='silentfix')

def cross_corr(f, g, shifts=None):
    """Cross correlation function.

    Accepts:
    f, g: array_like
    shifts: array_like

    Returns:
    c: array_like, correlation values."""
    if shifts is None:
        shifts = np.arange(len(g))
    c = np.zeros(len(shifts), dtype=float)
    f_pad = np.zeros(3*len(f))
    g_pad = np.zeros(3*len(f))
    f_pad[len(f): 2 * len(f)] = f - f.mean()
    g_pad[len(g): 2 * len(g)] = g - g.mean()
    for i in range(len(shifts)):
        c[i] = np.correlate(f_pad, np.roll(g_pad, -shifts[i]))

    return c

def wavelength_shift(in_pixel, ref_pixel, ref_wavelength, dispersion):

    """A helper function for determining wavelength shift.

    Accepts a pixel to be determined, a reference pixel, and the dispersion value."""

    pixel_diff = in_pixel - ref_pixel
    wavelength = pixel_diff * dispersion

    return wavelength

def norm_med_comb(data, region=((None, None), (None, None))):

    """Function implementing normalized median combination."""
    ((y1, x1), (y2, x2)) = region

    num_frames = data.shape[0]
    norm_factors = np.zeros(num_frames)
    norm_data = data.copy()

    for frame in range(num_frames):
        norm_factors[frame] = np.median(norm_data[frame, y1:y2, x1:x2])
        norm_data[frame] /= norm_factors[frame]

    med_comb_data = np.median(norm_data, axis = 0)

    return med_comb_data, norm_factors

def splinterp(x_new, x_old, y_old):

    """Function implementing spline interpolation."""

    spline = splrep(x_old, y_old)
    y_new = splev(x_new, spline)

    return y_new
