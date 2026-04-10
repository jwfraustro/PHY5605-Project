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
    """

    galaxy_median = np.median(galaxy_data, axis=1)

    left_edge_limit = 15
    right_edge_limit = len(galaxy_median) - 15

    galaxy_median[:left_edge_limit] = 0
    galaxy_median[right_edge_limit:] = 0

    zero_velocity_row = np.argmax(galaxy_median)

    save_plot(
        galaxy_median,
        filename="project_prob8_fig1.png",
        title="Galaxy Frame Median Y-Axis Pixel Values",
        xlabel="Y-Axis Pixel",
        ylabel="Median Value",
    )

    # By visual inspection of the plot, the full range of galaxy flux is between 50 and 250 pixels
    lower_flux_limit = 50
    upper_flux_limit = 250

    print(
        f"Galaxy Region Lower/Upper Flux Limits: {lower_flux_limit}, {upper_flux_limit}"
    )
    print(f"Galaxy Center 'Zero Velocity' Row: {zero_velocity_row}")

    return zero_velocity_row


def problem_9(galaxy_data: np.ndarray):
    """Problem 9:

    Define spectrum stripes and extract spectra.

    1. Divide the galaxy image into a series of horizontal stripes, 5 pixels high
        the middle of the galaxy must be in the middle vertical pixel of the stripe.
    2. Along each stripe, take the median of each 5-pixel stack, yielding a single 1D spectrum with no bad pixels.

    Plot this clean, smaller image. Find the row number of the galaxy center in this image.
    Print the number of rows in this reduced image and the index of the reference spectrum (which should be about half
    the number of rows).
    Plot the reference spectrum.
    """

    margin = 15
    stripe_size = 5
    y_size = galaxy_data.shape[0]
    x_size = galaxy_data.shape[1]
    n_stripes = y_size // stripe_size

    galaxy_data = galaxy_data[margin : y_size - margin, :]
    save_fits("FITS/galaxy_striped.fits", galaxy_data)

    galaxy_stripes = np.array_split(galaxy_data, n_stripes, axis=0)
    galaxy_stripes_median = np.array(
        [np.median(stripe, axis=0) for stripe in galaxy_stripes]
    )

    plt.imsave("new_plots/project_prob9_fig1.png", galaxy_stripes_median, cmap="gray")

    ref_row_idx = n_stripes // 2
    gal_spectrum = galaxy_stripes_median[ref_row_idx]

    plt.figure()
    plt.plot(gal_spectrum)
    plt.title("Reference Galaxy Spectrum")
    plt.xlabel("X-Axis Pixel")
    plt.ylabel("Median Value")
    plt.savefig("new_plots/project_prob9_fig2.png")

def main():
    """Run the reanalysis pipeline."""

    galaxy_data = fits.getdata(galaxy_corrected_fp)
    lamp_data = fits.getdata(lamp_corrected_fp)

    # Problem 8
    problem_8(galaxy_data)

    # Problem 9
    problem_9(galaxy_data)


if __name__ == "__main__":
    main()
