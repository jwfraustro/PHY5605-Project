"""A reanalysis of the 2019 data from the AST5765 project.

Because the original FITS files are no longer available, this reanalysis uses the intermediary data products from the
original run, which were resaved as FITS files. Because the issue with the original analysis was in the cross-correlation,
the reanalysis focuses on that step, and the subsequent steps that depend on it.
"""

import matplotlib

matplotlib.use("Agg")
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from cross_correlation import cross_correlate_subpixel
from project_funcs import save_fits

galaxy_corrected_fp = "FITS/galaxy_corrected.fits"
lamp_corrected_fp = "FITS/lamp_corrected.fits"

os.makedirs("new_plots", exist_ok=True)


def save_plot(
    data,
    filename: str = "temp_plot.png",
    title: str = "Plot",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
):
    """Helper function to save plot data."""
    out = Path("new_plots") / filename
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(out)
    print(f"Saved to {out}, exists: {out.exists()}")


def problem_8(galaxy_data: np.ndarray):
    """Problem 8:

    Find the y location of the galaxy's brightest peak. Take the median of the image along the horizontal axis,
    zero the messy edge regions, plot it, and find the row number of the peak. This roughly defines zero velocity.
    Use the plot to choose the upper and lower limits of the region with galaxy flux. Print these limits and the y-index
    of the row with the galaxy center.

    Returns
    -------
    zero_velocity_row : int
        Row index of the galaxy center (peak median flux).
    lower_flux_limit : int
        Lower row index of the region containing galaxy flux.
    upper_flux_limit : int
        Upper row index of the region containing galaxy flux.
    """

    galaxy_median = np.median(galaxy_data, axis=1)

    # Zero the messy edge regions so they don't confuse argmax.
    # Work on a copy so we don't touch the input array.
    median_masked = galaxy_median.copy()
    edge_margin = 15
    median_masked[:edge_margin] = 0
    median_masked[-(edge_margin):] = 0

    zero_velocity_row = np.argmax(median_masked)

    save_plot(
        median_masked,
        filename="problem_8_median_profile.png",
        title="Galaxy Frame Median Y-Axis Pixel Values",
        xlabel="Y-Axis Pixel",
        ylabel="Median Value",
    )

    # By visual inspection of the median profile, the galaxy flux region spans roughly rows 50–250.
    # These are used later to identify which stripes contain galaxy signal vs. sky.
    lower_flux_limit = 50
    upper_flux_limit = 250

    print(
        f"Galaxy Region Lower/Upper Flux Limits: {lower_flux_limit}, {upper_flux_limit}"
    )
    print(f"Galaxy Center 'Zero Velocity' Row: {zero_velocity_row}")

    return zero_velocity_row, lower_flux_limit, upper_flux_limit


def build_stripes(
    data: np.ndarray,
    center_row: int,
    stripe_size: int = 5,
) -> tuple[np.ndarray, int]:
    """Build median-collapsed stripes anchored on a center row.

    The center row is placed at the middle pixel of its stripe, and
    stripes extend outward in both directions to fill the frame.
    The frame is assumed to already be trimmed to usable rows.

    Parameters
    ----------
    data : np.ndarray
        2D frame (ny × nx), already trimmed to exclude bad edges.
    center_row : int
        Row index (in this frame's coordinates) that must land on
        the middle pixel of a stripe.
    stripe_size : int
        Height of each stripe in pixels.

    Returns
    -------
    stripes : np.ndarray
        2D array (n_stripes × nx) of median-collapsed spectra.
    ref_idx : int
        Index of the center stripe in the output array.
    """
    half = stripe_size // 2
    y_size, x_size = data.shape

    # How many full stripes fit above and below the center stripe?
    n_above = (center_row - half) // stripe_size
    n_below = (y_size - 1 - (center_row + half)) // stripe_size

    # Row bounds of the stripe grid
    top_row = center_row - half - n_above * stripe_size
    bot_row = center_row + half + n_below * stripe_size

    n_stripes = n_above + 1 + n_below
    ref_idx = n_above

    stripe_data = data[top_row : bot_row + 1, :]
    assert (
        stripe_data.shape[0] == n_stripes * stripe_size
    ), f"Expected {n_stripes * stripe_size} rows, got {stripe_data.shape[0]}"

    stripe_data = stripe_data.reshape(n_stripes, stripe_size, x_size)
    stripes = np.median(stripe_data, axis=1)

    return stripes, ref_idx


def problem_9(galaxy_trimmed: np.ndarray, zero_velocity_row: int):
    """Problem 9:

    Define spectrum stripes and extract spectra.

    The key constraint from the assignment: "The middle of the galaxy must
    be in the middle vertical pixel of a stripe." We anchor on the galaxy
    center row and build stripes outward from there.

    Parameters
    ----------
    galaxy_trimmed : np.ndarray
        The corrected 2D galaxy frame, already trimmed to exclude
        CCD edge garbage.
    zero_velocity_row : int
        Row index of the galaxy center in this (trimmed) frame.

    Returns
    -------
    galaxy_stripes : np.ndarray
        2D array of median-collapsed stripes (n_stripes × n_wavelengths).
    ref_row_idx : int
        Index of the reference (galaxy center) stripe in the output array.
    """

    galaxy_stripes, ref_row_idx = build_stripes(
        galaxy_trimmed, zero_velocity_row, stripe_size=5
    )
    n_stripes = galaxy_stripes.shape[0]

    print(f"Reduced image rows: {n_stripes}")
    print(f"Reference spectrum index: {ref_row_idx}")

    # Plot the reduced image
    plt.figure()
    plt.imshow(galaxy_stripes, cmap="gray", aspect="auto")
    plt.title("Reduced Galaxy Image (Median Stripes)")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Stripe Index")
    plt.savefig("new_plots/problem_9_striped_image.png")

    # Plot the reference spectrum
    gal_spectrum = galaxy_stripes[ref_row_idx]
    plt.figure()
    plt.plot(gal_spectrum)
    plt.title("Reference Galaxy Spectrum")
    plt.xlabel("X-Axis Pixel")
    plt.ylabel("Median Value")
    plt.savefig("new_plots/problem_9_ref_spectrum.png")

    return galaxy_stripes, ref_row_idx


def problem_10(
    galaxy_trimmed: np.ndarray,
    upper_flux_limit: int,
):
    """Problem 10:

    Extract galaxy background spectrum.

    Locate a region free of galaxy flux near the top of the slit (high row
    numbers). Take the column-wise median to get a 1-D sky spectrum, subtract
    it from a copy of the frame, and return both.

    Parameters
    ----------
    galaxy_trimmed : np.ndarray
        The corrected, trimmed 2D galaxy frame (not modified).
    upper_flux_limit : int
        Upper edge of the galaxy flux region, in trimmed-frame coordinates.

    Returns
    -------
    sky_spectrum : np.ndarray
        1-D sky background spectrum (one value per wavelength column).
    galaxy_sky_subtracted : np.ndarray
        Copy of galaxy_trimmed with sky_spectrum subtracted from every row.
    sky_low : int
        Low row bound of the sky region (trimmed-frame coords).
    sky_high : int
        High row bound of the sky region (exclusive, for slicing).
    """

    # The sky region runs from the upper edge of galaxy flux to the
    # end of the trimmed frame — all clean rows, no edge garbage.
    sky_low = upper_flux_limit
    sky_high = galaxy_trimmed.shape[0]

    sky_region = galaxy_trimmed[sky_low:sky_high, :]
    sky_spectrum = np.median(sky_region, axis=0)

    print(f"Sky region: rows {sky_low}–{sky_high - 1} ({sky_high - sky_low} rows)")

    # Plot the sky spectrum
    plt.figure()
    plt.plot(sky_spectrum)
    plt.title("Sky Background Spectrum")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Median Value")
    plt.savefig("new_plots/problem_10_sky_spectrum.png")

    # Subtract sky from a COPY of the galaxy frame
    galaxy_sky_subtracted = galaxy_trimmed.copy()
    galaxy_sky_subtracted -= sky_spectrum

    # Plot the sky-subtracted galaxy
    plt.figure()
    plt.imshow(galaxy_sky_subtracted, cmap="gray", aspect="auto", origin="lower")
    plt.title("Galaxy After Initial Sky Subtraction")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Row")
    plt.savefig("new_plots/problem_10_sky_subtracted.png")

    return sky_spectrum, galaxy_sky_subtracted, sky_low, sky_high


def problem_11(
    lamp_trimmed: np.ndarray,
    zero_velocity_row: int,
):
    """Problem 11:

    Extract lamp spectra for intrinsic shift measurement.

    The lamp frame has already been trimmed to match the galaxy frame's
    coordinate system (see main()), so the galaxy center row index
    applies directly.

    Parameters
    ----------
    lamp_trimmed : np.ndarray
        The corrected, trimmed 2D lamp frame.
    zero_velocity_row : int
        Galaxy center row in trimmed-frame coordinates.

    Returns
    -------
    lamp_stripes : np.ndarray
        2D array of median-collapsed lamp stripes.
    lamp_ref_idx : int
        Index of the lamp stripe corresponding to the galaxy center.
    """

    lamp_stripes, lamp_ref_idx = build_stripes(
        lamp_trimmed, zero_velocity_row, stripe_size=5
    )

    print(
        f"Lamp stripes: {lamp_stripes.shape[0]} stripes from "
        f"{lamp_trimmed.shape[0]} rows"
    )
    print(f"Lamp reference stripe index: {lamp_ref_idx}")

    # Plot the striped lamp image
    plt.figure()
    plt.imshow(lamp_stripes, cmap="gray", aspect="auto")
    plt.title("Reduced Lamp Image (Median Stripes)")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Stripe Index")
    plt.savefig("new_plots/problem_11_lamp_stripes.png")

    # Plot the lamp spectrum at the galaxy center
    lamp_spectrum = lamp_stripes[lamp_ref_idx]
    plt.figure()
    plt.plot(lamp_spectrum)
    plt.title("Lamp Spectrum at Galaxy Center")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Value")
    plt.savefig("new_plots/problem_11_lamp_ref_spectrum.png")

    return lamp_stripes, lamp_ref_idx


def problem_12(
    lamp_stripes: np.ndarray,
    lamp_ref_idx: int,
    galaxy_stripes: np.ndarray,
    galaxy_ref_idx: int,
):
    """Problem 12:

    Calculate intrinsic shifts from the lamp, and raw shifts from the galaxy.

    For the lamp: each stripe is cross-correlated against the lamp reference
    spectrum (at the galaxy center row). These shifts measure the optical
    distortion of the Double Spectrograph — constant-wavelength lines that
    curve slightly across the slit. These are a systematic error to be
    removed from the galaxy shifts.

    For the galaxy: each stripe is cross-correlated against the galaxy
    reference spectrum. These shifts contain both the intrinsic distortion
    AND the Doppler signal from galactic rotation. Separating the two
    comes in Problem 15.

    Parameters
    ----------
    lamp_stripes : np.ndarray
        2D array of median-collapsed lamp spectra (n_stripes × n_wavelengths).
    lamp_ref_idx : int
        Index of the lamp reference stripe.
    galaxy_stripes : np.ndarray
        2D array of median-collapsed galaxy spectra.
    galaxy_ref_idx : int
        Index of the galaxy reference stripe.

    Returns
    -------
    lamp_shifts : np.ndarray
        Intrinsic (optical distortion) shift for each lamp stripe,
        in original pixel units.
    lamp_shift_errs : np.ndarray
        Uncertainties on the lamp shifts.
    galaxy_shifts : np.ndarray
        Raw shift for each galaxy stripe relative to the galaxy
        reference spectrum, in original pixel units.
    galaxy_shift_errs : np.ndarray
        Uncertainties on the galaxy shifts.
    """

    lamp_ref_spectrum = lamp_stripes[lamp_ref_idx]
    galaxy_ref_spectrum = galaxy_stripes[galaxy_ref_idx]

    n_lamp = lamp_stripes.shape[0]
    n_galaxy = galaxy_stripes.shape[0]

    # --- Lamp shifts (intrinsic distortion) ---
    # The lamp lines are constant-wavelength, so any measured shift is
    # purely instrumental. Shift range of ±10 pixels is generous for
    # optical distortion which is typically sub-pixel to a few pixels.
    lamp_shifts = np.zeros(n_lamp)
    lamp_shift_errs = np.zeros(n_lamp)

    print(f"Calculating lamp intrinsic shifts ({n_lamp} stripes)...")
    for i in range(n_lamp):
        shift, err, _, _ = cross_correlate_subpixel(
            lamp_stripes[i],
            lamp_ref_spectrum,
            shift_range=(-10, 10),
            supersample=10,
        )
        lamp_shifts[i] = shift
        lamp_shift_errs[i] = err

    # Plot lamp shifts
    plt.figure()
    plt.plot(lamp_shifts, ".-")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.title("Lamp Intrinsic Shifts (Optical Distortion)")
    plt.xlabel("Stripe Index")
    plt.ylabel("Shift (pixels)")
    plt.savefig("new_plots/problem_12_lamp_shifts.png")

    print(f"  Lamp shift range: {lamp_shifts.min():.3f} to {lamp_shifts.max():.3f} px")
    print(
        f"  Lamp shift at reference: {lamp_shifts[lamp_ref_idx]:.6f} px (should be ~0)"
    )

    # --- Galaxy shifts (Doppler + intrinsic) ---
    # The galaxy shifts contain both the rotation signal (the S-curve we
    # want) and the intrinsic distortion (which we'll subtract in Problem 15).
    # Shift range of ±10 pixels is adequate for the expected Doppler shifts.
    galaxy_shifts = np.zeros(n_galaxy)
    galaxy_shift_errs = np.zeros(n_galaxy)

    print(f"Calculating galaxy shifts ({n_galaxy} stripes)...")
    for i in range(n_galaxy):
        shift, err, _, _ = cross_correlate_subpixel(
            galaxy_stripes[i],
            galaxy_ref_spectrum,
            shift_range=(-10, 10),
            supersample=10,
        )
        galaxy_shifts[i] = shift
        galaxy_shift_errs[i] = err

    # Plot galaxy shifts
    plt.figure()
    plt.plot(galaxy_shifts, ".-")
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.title("Galaxy Raw Shifts (Doppler + Intrinsic)")
    plt.xlabel("Stripe Index")
    plt.ylabel("Shift (pixels)")
    plt.savefig("new_plots/problem_12_galaxy_shifts.png")

    print(
        f"  Galaxy shift range: {galaxy_shifts.min():.3f} to {galaxy_shifts.max():.3f} px"
    )
    print(
        f"  Galaxy shift at reference: {galaxy_shifts[galaxy_ref_idx]:.6f} px (should be ~0)"
    )

    return lamp_shifts, lamp_shift_errs, galaxy_shifts, galaxy_shift_errs


def problem_13(
    galaxy_trimmed: np.ndarray,
    sky_spectrum: np.ndarray,
    lamp_shifts: np.ndarray,
    lamp_ref_idx: int,
    sky_low: int,
    sky_high: int,
    zero_velocity_row: int,
    stripe_size: int = 5,
):
    """Problem 13:

    Shift and remove background spectra.

    For each row of the galaxy frame, determine the intrinsic shift at
    that slit position (interpolated from the lamp shift curve), compute
    the difference relative to the average sky-region shift, shift the
    sky spectrum by that amount using spline interpolation, and subtract.

    This removes sky lines much more cleanly than the naive subtraction
    in Problem 10, because it accounts for the optical distortion that
    makes sky lines curve slightly across the slit.

    Parameters
    ----------
    galaxy_trimmed : np.ndarray
        The corrected, trimmed 2D galaxy frame (not modified).
    sky_spectrum : np.ndarray
        1-D sky background spectrum from Problem 10.
    lamp_shifts : np.ndarray
        Per-stripe intrinsic shifts from Problem 12.
    lamp_ref_idx : int
        Index of the lamp reference stripe.
    sky_low, sky_high : int
        Row bounds of the sky region (trimmed-frame coords, sky_high exclusive).
    zero_velocity_row : int
        Galaxy center row in trimmed-frame coords.
    stripe_size : int
        Stripe height used in build_stripes.

    Returns
    -------
    galaxy_cleaned : np.ndarray
        Sky-subtracted galaxy frame with intrinsic-shift-corrected sky removal.
    """
    from project_funcs import splinterp

    y_size, x_size = galaxy_trimmed.shape

    # The lamp_shifts array is per-stripe. We need per-row shifts, so
    # interpolate the lamp shift curve to every row in the trimmed frame.
    #
    # Each stripe i corresponds to a range of rows centered on:
    #   row = zero_velocity_row + (i - lamp_ref_idx) * stripe_size
    # (since stripes are anchored on zero_velocity_row at lamp_ref_idx)
    n_lamp_stripes = len(lamp_shifts)
    stripe_centers = (
        zero_velocity_row + (np.arange(n_lamp_stripes) - lamp_ref_idx) * stripe_size
    )
    row_indices = np.arange(y_size, dtype=float)

    # Interpolate the lamp shift curve to every row
    # Clamp to the range of stripe centers to avoid extrapolation artifacts
    lamp_shift_per_row = np.interp(row_indices, stripe_centers, lamp_shifts)

    # Average intrinsic shift in the sky region
    sky_rows = np.arange(sky_low, sky_high, dtype=float)
    avg_sky_shift = np.mean(np.interp(sky_rows, stripe_centers, lamp_shifts))

    print(f"Average intrinsic shift in sky region: {avg_sky_shift:.4f} px")

    # For each row, shift the sky spectrum by the difference between
    # that row's intrinsic shift and the sky region's average shift,
    # then subtract.
    galaxy_cleaned = galaxy_trimmed.copy()
    x_orig = np.arange(x_size, dtype=float)

    for i in range(y_size):
        shift_diff = lamp_shift_per_row[i] - avg_sky_shift
        # Shift the sky spectrum: evaluate it at (x - shift_diff)
        # If shift_diff > 0, the sky at this row is shifted right relative
        # to the sky region, so we evaluate the sky at x - shift_diff
        sky_shifted = splinterp(x_orig - shift_diff, x_orig, sky_spectrum)
        galaxy_cleaned[i] -= sky_shifted

    # Clean up the bad pixels
    for i, rows in enumerate(galaxy_cleaned):
        for j, pixels in enumerate(rows):
            if galaxy_cleaned[i, j] > 200:
                galaxy_cleaned[i, j] = np.mean(
                    [
                        galaxy_cleaned[i, j + 3],
                        galaxy_cleaned[i, j - 3],
                        galaxy_cleaned[i - 3, j],
                        galaxy_cleaned[i + 3, j],
                    ]
                )

    # Plot the cleaned galaxy
    plt.figure()
    plt.imshow(galaxy_cleaned, cmap="gray", aspect="auto", origin="lower")
    plt.title("Galaxy After Intrinsic-Shift-Corrected Sky Subtraction")
    plt.xlabel("Wavelength Pixel")
    plt.ylabel("Row")
    plt.savefig("new_plots/problem_13_galaxy_cleaned.png")

    return galaxy_cleaned


def problem_14(
    galaxy_cleaned: np.ndarray,
    zero_velocity_row: int,
    lower_flux_limit: int,
    upper_flux_limit: int,
):
    """Problem 14:

    Find galactic-rotation Doppler shifts from the cleaned galaxy frame.

    Re-stripe the cleaned frame and cross-correlate each stripe against
    the reference spectrum. Identify the stripes with meaningful galaxy
    signal and trim the rest.

    Parameters
    ----------
    galaxy_cleaned : np.ndarray
        Sky-subtracted galaxy frame from Problem 13.
    zero_velocity_row : int
        Galaxy center row in trimmed-frame coords.
    lower_flux_limit : int
        Lower edge of galaxy flux region (trimmed-frame coords).
    upper_flux_limit : int
        Upper edge of galaxy flux region (trimmed-frame coords).

    Returns
    -------
    doppler_shifts : np.ndarray
        Per-stripe Doppler shifts for ALL stripes (including non-galaxy).
    doppler_shift_errs : np.ndarray
        Uncertainties on the Doppler shifts.
    good_mask : np.ndarray
        Boolean mask indicating which stripes contain galaxy signal.
    cleaned_stripes : np.ndarray
        The re-striped cleaned galaxy image.
    cleaned_ref_idx : int
        Reference stripe index in the cleaned stripes.
    """

    # Re-stripe the cleaned galaxy frame
    cleaned_stripes, cleaned_ref_idx = build_stripes(
        galaxy_cleaned, zero_velocity_row, stripe_size=5
    )
    n_stripes = cleaned_stripes.shape[0]
    ref_spectrum = cleaned_stripes[cleaned_ref_idx]

    print(f"Re-striped cleaned galaxy: {n_stripes} stripes, ref at {cleaned_ref_idx}")

    # Cross-correlate each stripe against the reference
    doppler_shifts = np.zeros(n_stripes)
    doppler_shift_errs = np.zeros(n_stripes)

    print(f"Calculating Doppler shifts ({n_stripes} stripes)...")
    for i in range(n_stripes):
        shift, err, _, _ = cross_correlate_subpixel(
            cleaned_stripes[i],
            ref_spectrum,
            shift_range=(-10, 10),
            supersample=10,
        )
        doppler_shifts[i] = shift
        doppler_shift_errs[i] = err

    # Determine which stripes contain galaxy flux.
    # Convert flux limits from row coordinates to stripe indices.
    stripe_size = 5
    half = stripe_size // 2
    # The stripe grid starts at row: zero_velocity_row - half - cleaned_ref_idx * stripe_size
    top_row_of_grid = zero_velocity_row - half - cleaned_ref_idx * stripe_size

    # Stripe i spans rows [top_row_of_grid + i*5, top_row_of_grid + i*5 + 4]
    # A stripe is "good" if its center falls within the galaxy flux region
    stripe_center_rows = top_row_of_grid + np.arange(n_stripes) * stripe_size + half
    good_mask = (stripe_center_rows >= lower_flux_limit) & (
        stripe_center_rows <= upper_flux_limit
    )

    n_good = good_mask.sum()
    print(f"Stripes with galaxy flux: {n_good} of {n_stripes}")

    # Plot all shifts
    plt.figure()
    plt.plot(doppler_shifts, ".-", alpha=0.5, label="All stripes")
    plt.plot(
        np.where(good_mask)[0], doppler_shifts[good_mask], ".-", label="Galaxy flux"
    )
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.title("Doppler Shifts (All Stripes)")
    plt.xlabel("Stripe Index")
    plt.ylabel("Shift (pixels)")
    plt.savefig("new_plots/problem_14_doppler_shifts_all.png")

    # Plot trimmed (good) shifts centered on galaxy center
    good_indices = np.where(good_mask)[0]
    good_offsets = good_indices - cleaned_ref_idx  # distance from center in stripes
    plt.figure()
    plt.errorbar(
        good_offsets,
        doppler_shifts[good_mask],
        yerr=doppler_shift_errs[good_mask],
        fmt=".-",
        capsize=2,
    )
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.title("Doppler Shifts (Galaxy Flux Region)")
    plt.xlabel("Stripe Offset from Galaxy Center")
    plt.ylabel("Shift (pixels)")
    plt.savefig("new_plots/problem_14_doppler_shifts_trimmed.png")

    return (
        doppler_shifts,
        doppler_shift_errs,
        good_mask,
        cleaned_stripes,
        cleaned_ref_idx,
    )


def problem_15(
    doppler_shifts: np.ndarray,
    doppler_shift_errs: np.ndarray,
    lamp_shifts: np.ndarray,
    good_mask: np.ndarray,
    cleaned_ref_idx: int,
):
    """Problem 15:

    Remove systematic (intrinsic) shifts from the Doppler shifts.

    The lamp shifts measured in Problem 12 capture the optical distortion.
    Subtracting them from the raw Doppler shifts isolates the true
    galactic rotation signal.

    Note: the lamp and galaxy stripe grids are anchored on the same center
    row, so stripe indices correspond directly. If the stripe counts differ
    (because the lamp frame is slightly different in size), we align on
    the reference index.

    Parameters
    ----------
    doppler_shifts : np.ndarray
        Per-stripe Doppler shifts from Problem 14.
    doppler_shift_errs : np.ndarray
        Uncertainties on the Doppler shifts.
    lamp_shifts : np.ndarray
        Per-stripe intrinsic shifts from Problem 12.
    good_mask : np.ndarray
        Boolean mask for stripes with galaxy signal.
    cleaned_ref_idx : int
        Reference stripe index (galaxy center) from Problem 14.

    Returns
    -------
    corrected_shifts : np.ndarray
        Doppler shifts with intrinsic distortion removed (all stripes).
    corrected_shift_errs : np.ndarray
        Uncertainties on the corrected shifts (same as input — the lamp
        shift uncertainty is negligible compared to the galaxy shift
        uncertainty).
    """

    n_doppler = len(doppler_shifts)
    n_lamp = len(lamp_shifts)

    # Both stripe grids were built with build_stripes() anchored on the same
    # center row and stripe size, from frames of equal height (galaxy and lamp
    # were trimmed to match in main()). So they have the same number of stripes
    # and stripe indices correspond directly.
    if n_lamp == n_doppler:
        intrinsic_at_doppler = lamp_shifts
    else:
        # Shouldn't happen with properly aligned frames, but handle gracefully
        intrinsic_at_doppler = np.interp(
            np.linspace(0, n_lamp - 1, n_doppler),
            np.arange(n_lamp, dtype=float),
            lamp_shifts,
        )

    corrected_shifts = doppler_shifts - intrinsic_at_doppler
    corrected_shift_errs = doppler_shift_errs  # lamp uncertainty is negligible

    # The shift at the galaxy center should be ~0 (it's the reference)
    print(
        f"Corrected shift at galaxy center: {corrected_shifts[cleaned_ref_idx]:.6f} px (should be ~0)"
    )

    # Plot corrected shifts for the good region
    good_indices = np.where(good_mask)[0]
    good_offsets = good_indices - cleaned_ref_idx

    plt.figure()
    plt.errorbar(
        good_offsets,
        corrected_shifts[good_mask],
        yerr=corrected_shift_errs[good_mask],
        fmt=".-",
        capsize=2,
        label="Corrected (Doppler only)",
    )
    plt.plot(
        good_offsets,
        doppler_shifts[good_mask],
        "x",
        alpha=0.3,
        label="Raw (before correction)",
    )
    plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.title("Doppler Shifts Corrected for Intrinsic Distortion")
    plt.xlabel("Stripe Offset from Galaxy Center")
    plt.ylabel("Shift (pixels)")
    plt.savefig("new_plots/problem_15_corrected_shifts.png")

    return corrected_shifts, corrected_shift_errs


def problem_17(
    lamp_spectrum: np.ndarray,
    line_1_range: tuple[int, int] = (36, 47),
    line_2_range: tuple[int, int] = (1013, 1024),
    line_1_wavelength: float = 6532.8962,
    line_2_wavelength: float = 7173.9104,
) -> tuple[float, float, float]:
    """Problem 17: Calculate dispersion from two identified lamp lines.

    Fits a Gaussian to each line to find sub-pixel centers, then
    computes dispersion = Δλ / Δpixel.

    Returns
    -------
    dispersion : float
        Angstroms per pixel.
    line_1_center : float
        Sub-pixel center of line 1.
    line_2_center : float
        Sub-pixel center of line 2.
    """
    from gaussian import fitgaussian

    l1_start, l1_end = line_1_range
    l2_start, l2_end = line_2_range

    line_1_center = fitgaussian(lamp_spectrum[l1_start:l1_end])[1][0] + l1_start
    line_2_center = fitgaussian(lamp_spectrum[l2_start:l2_end])[1][0] + l2_start

    dispersion = (line_2_wavelength - line_1_wavelength) / (
        line_2_center - line_1_center
    )

    print(f"Line 1 center: {line_1_center:.3f} px  ({line_1_wavelength:.4f} Å)")
    print(f"Line 2 center: {line_2_center:.3f} px  ({line_2_wavelength:.4f} Å)")
    print(f"Dispersion: {dispersion:.6f} Å/px")

    return dispersion, line_1_center, line_2_center


def main():
    """Run the reanalysis pipeline."""

    galaxy_raw = fits.getdata(galaxy_corrected_fp).astype(np.float64)
    lamp_raw = fits.getdata(lamp_corrected_fp).astype(np.float64)

    # Problem 8: find galaxy center and flux limits (operates on full frame)
    zero_velocity_row, lower_flux, upper_flux = problem_8(galaxy_raw)

    # Trim both frames to exclude CCD edge garbage.
    # The galaxy frame needs 15 rows removed from each end.
    # The lamp was already trimmed by 15 rows on each end during Problem 7
    # (lamp_data = lamp_data[15:-15] before saving lamp_corrected.fits),
    # so it's already 30 rows shorter. We trim the galaxy to match.
    edge_margin = 15
    galaxy_data = galaxy_raw[edge_margin:-edge_margin, :]
    # The lamp's saved trim matches the galaxy's trim, so their row
    # indices now correspond to the same physical slit positions.
    lamp_data = lamp_raw

    # Adjust row indices from full-frame to trimmed-frame coordinates
    zero_velocity_row -= edge_margin
    lower_flux -= edge_margin
    upper_flux -= edge_margin

    print(f"\nTrimmed galaxy frame: {galaxy_data.shape} (was {galaxy_raw.shape})")
    print(f"Lamp frame: {lamp_data.shape} (pre-trimmed during Problem 7)")
    print(f"Adjusted galaxy center row: {zero_velocity_row}")

    # Problem 9: stripe the galaxy frame, anchored on galaxy center
    galaxy_stripes, ref_row_idx = problem_9(galaxy_data, zero_velocity_row)

    # Problem 10: extract and subtract sky background
    sky_spectrum, galaxy_sky_sub, sky_low, sky_high = problem_10(
        galaxy_data, upper_flux
    )

    # Problem 11: stripe the lamp frame for intrinsic shift measurement
    lamp_stripes, lamp_ref_idx = problem_11(lamp_data, zero_velocity_row)

    # Problem 12: calculate intrinsic shifts (lamp) and raw shifts (galaxy)
    lamp_shifts, lamp_shift_errs, galaxy_shifts, galaxy_shift_errs = problem_12(
        lamp_stripes, lamp_ref_idx, galaxy_stripes, ref_row_idx
    )

    # Problem 13: shift-corrected sky subtraction
    galaxy_cleaned = problem_13(
        galaxy_data,
        sky_spectrum,
        lamp_shifts,
        lamp_ref_idx,
        sky_low,
        sky_high,
        zero_velocity_row,
    )

    save_fits("foo.fits", galaxy_cleaned)

    # Problem 14: find Doppler shifts from the cleaned galaxy
    (
        doppler_shifts,
        doppler_errs,
        good_mask,
        cleaned_stripes,
        cleaned_ref_idx,
    ) = problem_14(
        galaxy_cleaned,
        zero_velocity_row,
        lower_flux,
        upper_flux,
    )

    # Problem 15: remove intrinsic shifts from Doppler shifts
    corrected_shifts, corrected_errs = problem_15(
        doppler_shifts,
        doppler_errs,
        lamp_shifts,
        good_mask,
        cleaned_ref_idx,
    )

    # Problem 17: calculate dispersion from lamp lines
    lamp_ref_spectrum = lamp_stripes[lamp_ref_idx]
    dispersion, line_1_center, line_2_center = problem_17(lamp_ref_spectrum)


if __name__ == "__main__":
    main()
