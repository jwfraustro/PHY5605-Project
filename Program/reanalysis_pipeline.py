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


def problem_9(galaxy_data: np.ndarray, zero_velocity_row: int):
    """Problem 9:

    Define spectrum stripes and extract spectra.

    The key constraint from the assignment: "The middle of the galaxy must be in the middle vertical pixel of a stripe."
    So we anchor on the galaxy center row and build stripes outward from there.

    Parameters
    ----------
    galaxy_data : np.ndarray
        The corrected 2D galaxy frame.
    zero_velocity_row : int
        Row index of the galaxy center from problem 8.

    Returns
    -------
    galaxy_stripes : np.ndarray
        2D array of median-collapsed stripes (n_stripes x n_wavelengths).
    ref_row_idx : int
        Index of the reference (galaxy center) stripe in the output array.
    """

    stripe_size = 5
    half_stripe = stripe_size // 2  # = 2, so center pixel is index 2 within each stripe
    y_size, x_size = galaxy_data.shape

    # The CCD edges have bad pixels — skip them. The assignment says
    # "eliminate the messy edges of the frame, but keep as much sky as you can."
    # We define a safe region and only build stripes within it.
    edge_margin = 15
    safe_top = edge_margin  # first usable row
    safe_bot = y_size - edge_margin - 1  # last usable row

    # The galaxy center row must be the middle pixel of its stripe.
    # That means the stripe runs from (center - 2) to (center + 2) inclusive.
    # Every other stripe is offset by multiples of stripe_size from there.
    n_above = (zero_velocity_row - half_stripe - safe_top) // stripe_size
    n_below = (safe_bot - (zero_velocity_row + half_stripe)) // stripe_size

    # Top row of the topmost stripe
    top_row = zero_velocity_row - half_stripe - n_above * stripe_size
    # Bottom row of the bottommost stripe (inclusive)
    bot_row = zero_velocity_row + half_stripe + n_below * stripe_size

    n_stripes = n_above + 1 + n_below  # +1 for the center stripe
    ref_row_idx = n_above  # center stripe's index in the output

    # Extract the usable portion of the frame and reshape into stripes
    stripe_data = galaxy_data[top_row : bot_row + 1, :]
    assert (
        stripe_data.shape[0] == n_stripes * stripe_size
    ), f"Expected {n_stripes * stripe_size} rows, got {stripe_data.shape[0]}"

    # Reshape to (n_stripes, stripe_size, x_size) and median-collapse each stripe
    stripe_data = stripe_data.reshape(n_stripes, stripe_size, x_size)
    galaxy_stripes = np.median(stripe_data, axis=1)

    # Sanity checks
    print(
        f"Stripe grid: rows {top_row}–{bot_row} of {y_size}, "
        f"{n_stripes} stripes of {stripe_size} px"
    )
    print(f"Reduced image rows: {n_stripes}")
    print(f"Reference spectrum index: {ref_row_idx}")
    print(
        f"Center row {zero_velocity_row} sits at pixel {half_stripe} "
        f"(middle) of stripe {ref_row_idx}"
    )

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


def main():
    """Run the reanalysis pipeline."""

    galaxy_data = fits.getdata(galaxy_corrected_fp).astype(np.float64)
    lamp_data = fits.getdata(lamp_corrected_fp).astype(np.float64)

    # Problem 8
    zero_velocity_row, lower_flux, upper_flux = problem_8(galaxy_data)

    # Problem 9
    galaxy_stripes, ref_row_idx = problem_9(galaxy_data, zero_velocity_row)


if __name__ == "__main__":
    main()
