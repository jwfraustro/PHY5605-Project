#! /usr/bin/env python3
# Joshua Fraustro - joshua.fraustro@gmail.com
# AST5765 Astronomical Data Analysis
# Final Project

import numpy as np
import gaussian as g
from math import floor
from gaussian import fitgaussian
from project_funcs import splinterp, save_fits, norm_med_comb, cross_corr, wavelength_shift
from matplotlib import pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clip
import random

plt.ion()

data_dir = '../../WebCourses/Files/data/galaxy/data/'
flat_files = [
    'bs324n.fits',
    'bs325n.fits',
    'bs326n.fits',
    'bs327n.fits',
    'bs328n.fits',
    'bs329n.fits',
    'bs330n.fits',
    'bs331n.fits',
    'bs332n.fits',
    'bs333n.fits'
]
bias_files = [
    'bs334n.fits',
    'bs335n.fits',
    'bs336n.fits',
    'bs337n.fits',
    'bs338n.fits',
    'bs339n.fits',
    'bs340n.fits',
    'bs341n.fits',
    'bs342n.fits',
    'bs343n.fits'
]

galaxy_files = ['bs397n.fits']
lamp_files = ['bs395n.fits']

# Problem 1
print("Problem 1")

mag = 5.
lo = -2.
hi = 2.
shift = -0.2

xold = np.linspace(lo, hi, np.round(hi - lo + 1))
xnew = np.linspace(lo, hi, np.round((hi - lo) * mag + 1)) + shift
yold = g.gaussian(xold)
ynew = splinterp(xnew, xold, yold)
print(ynew)

# Problem 2
print("Problem 2")
print("A short test of the interpolation function.")
xold = np.linspace(-3, 3)
yold = np.sin(xold)
xnew = np.linspace(-3+2, 3+2, len(xold)*10)
ynew = splinterp(xnew, xold, yold)

# Problem 3
print("\nProblem 3")
# c = cross_corr(yold, ynew, xnew)

# Problem 4
print("\nProblem 4")

# Problem 5
print("\nProblem 5")
temp_frame = fits.getdata(data_dir + bias_files[0])
temp_frame_y, temp_frame_x = temp_frame.shape
num_bias_frames = len(bias_files)

bias_frame_data = np.zeros((num_bias_frames, temp_frame_y, temp_frame_x), dtype=float)

for frame in range(num_bias_frames):
    file_name = data_dir + bias_files[frame]
    bias_frame_data[frame] = fits.getdata(file_name)

bias_med_comb = np.median(bias_frame_data, axis=0)
save_fits('bias_med_comb.fits', bias_frame_data)

print("Master bias frame pixel (150,150) value:", bias_med_comb[150, 150])

# Problem 6
print("\nProblem 6")

temp_frame = fits.getdata(data_dir + flat_files[0])
temp_frame_y, temp_frame_x = temp_frame.shape
num_flat_frames = len(flat_files)

flat_frame_data = np.zeros((num_flat_frames, temp_frame_y, temp_frame_x), dtype=float)

for frame in range(num_bias_frames):
    file_name = data_dir + flat_files[frame]
    flat_frame_data[frame] = fits.getdata(file_name)

flat_frame_data -= bias_med_comb

master_dome_flat, dome_flat_norms = norm_med_comb(flat_frame_data)
print("Flat field image pixel (150,150) value: ", master_dome_flat[150, 150])
save_fits('master_dome_flat.fits', master_dome_flat)

# Problem 7
print("\nProblem 7")

galaxy_data = np.float32(fits.getdata(data_dir + galaxy_files[0]))
lamp_data = np.float32(fits.getdata(data_dir + lamp_files[0]))

save_fits('galaxy_uncorrected.fits', galaxy_data)
plt.imsave('galaxy_uncorrected.png', galaxy_data, cmap='gray')
# save_fits('lamp_uncorrected.fits', lamp_data)

galaxy_data -= bias_med_comb
lamp_data -= bias_med_comb

galaxy_data /= master_dome_flat
lamp_data /= master_dome_flat

lamp_data = lamp_data[15:-15]

save_fits('galaxy_corrected.fits', galaxy_data)
save_fits('lamp_corrected.fits', lamp_data)

print("Galaxy frame pixel (150,150) value:", galaxy_data[150, 150])

# Problem 8
print("\nProblem 8")
galaxy_median = np.median(galaxy_data, axis=1)

galaxy_median[0:12] = 0.
galaxy_median[-13:] = 0.

plt.figure()
plt.title("Galaxy Frame Median Y-Axis Pixel Values")
plt.ylabel("Median Value")
plt.xlabel("Y-Axis Pixel")
plt.plot(galaxy_median)
plt.savefig("project_prob8_fig1.png")

zero_velocity_row = np.argmax(galaxy_median)

lower_flux_limit = 67
upper_flux_limit = 220

print("Galaxy Region Lower/Upper Flux Limits: %i, %i" % (lower_flux_limit, upper_flux_limit))
print("Galaxy Center 'Zero Velocity' Row:", zero_velocity_row)

# Problem 9
print("\nProblem 9")

margin = 15
stripe_size = 5
y_size = galaxy_data.shape[0]
x_size = galaxy_data.shape[1]
n_stripes = y_size // 5

galaxy_data = galaxy_data[margin:y_size - margin, :]
# save_fits('galaxy_trim.fits',galaxy_frame_data)

galaxy_stripes = np.array_split(galaxy_data, n_stripes, axis=0)
for stripe in range(len(galaxy_stripes)):
    galaxy_stripes[stripe] = np.median(galaxy_stripes[stripe], axis=0)

plt.imsave('project_prob9_fig1.png', galaxy_stripes, cmap='gray')

ref_row_num = n_stripes // 2
gal_spectrum = galaxy_stripes[ref_row_num]
print("Reduced image rows:", len(galaxy_stripes))
print("Reference spectrum index:", ref_row_num)

plt.clf()
plt.title('Galaxy Reference Spectrum Values')
plt.xlabel('Image X-Position')
plt.ylabel('Values')
plt.plot(gal_spectrum)
plt.savefig('project_prob9_fig2.png')

# Problem 10
print("\nProblem 10")
skyreg_low_bound = 240
skyreg_upper_bound = y_size

skyreg = galaxy_data[skyreg_low_bound:skyreg_upper_bound, :]
sky_spectrum = np.median(skyreg, axis=0)

# save_fits('sky_region.fits', skyreg)

print("Sky region low row:", skyreg_low_bound)
print("Sky region upper row:", skyreg_upper_bound)

plt.clf()
plt.title("Sky Sample Spectrum")
plt.xlabel("Image X-Position")
plt.ylabel("Values")
plt.plot(sky_spectrum)
plt.savefig('project_prob10_fig1.png')

gal_frame_copy = galaxy_data.copy()
gal_frame_copy -= sky_spectrum

save_fits('galaxy_sub_sky.fits', gal_frame_copy)
plt.imsave('project_prob10_fig1.png', gal_frame_copy, cmap='gray')

# Problem 11
print("\nProblem 11")

print("Creating intrinsic shift lamp spectra")
y_size = lamp_data.shape[0]
num_stripes = y_size // stripe_size
lamp_stripes = np.array_split(lamp_data, num_stripes, axis=0)

for stripe in range(len(lamp_stripes)):
    lamp_stripes[stripe] = np.median(lamp_stripes[stripe], axis=0)

plt.imsave('project_prob11_figure1.png', lamp_stripes, cmap='gray')

gal_center_row = len(lamp_stripes) // 2
lamp_spectrum = lamp_stripes[gal_center_row]

plt.clf()
plt.title("Lamp Spectrum at Galaxy Center")
plt.xlabel("Image X-Position")
plt.ylabel("Value")
plt.plot(lamp_spectrum)
plt.savefig('project_prob11_figure2.png')

# Problem 12
print("\nProblem 12")

res_factor = 10
shifts = range(-100,100)

x_old_range = np.arange(len(lamp_spectrum))
x_new_range = np.linspace(np.min(x_old_range), np.max(x_old_range), len(lamp_spectrum) * res_factor)

interp_lamp_spectrum = splinterp(x_new_range, x_old_range, lamp_spectrum)

lamp_shifts = []
print("Calculating Lamp Spectrum Shifts")
for slice in lamp_stripes:
    interp_slice = splinterp(x_new_range, x_old_range, slice)
    c = cross_corr(interp_slice, interp_lamp_spectrum, shifts)
    lamp_shifts.append(shifts[np.argmax(c)])

plt.clf()
plt.title("Lamp Spectrum Shifts")
plt.xlabel("Spectrum Slice")
plt.ylabel("Shift Amount")
plt.plot(lamp_shifts)
plt.savefig('lamp_spectrum_shifts.png')

x_old_range = np.arange(len(gal_spectrum))
x_new_range = np.linspace(np.min(x_old_range), np.max(x_old_range), len(gal_spectrum) * res_factor)

interp_gal_spectrum = splinterp(x_new_range, x_old_range, gal_spectrum)

galaxy_shifts = []
print("Calculating Galaxy Spectrum Shifts")
for slice in galaxy_stripes:
    interp_slice = splinterp(x_new_range, x_old_range, slice)
    c = cross_corr(interp_slice, interp_gal_spectrum, shifts)
    galaxy_shifts.append(shifts[np.argmax(c)])

plt.clf()
plt.title("Galaxy Spectrum Shifts")
plt.xlabel("Spectrum Slice")
plt.ylabel("Shift Amount")
plt.plot(galaxy_shifts)
plt.savefig('galaxy_spectrum_shifts.png')

# Problem 13
print("\nProblem 13")

sky_shifts = []

for slice in galaxy_data[skyreg_low_bound:skyreg_upper_bound]:
    interp_slice = splinterp(x_new_range, x_old_range, slice)
    c = cross_corr(interp_slice, interp_lamp_spectrum, shifts)
    sky_shifts.append(shifts[np.argmax(c)])

avg_shift = np.mean(sky_shifts)

correction_shifts = []

interp_sky_spec = splinterp(x_new_range, x_old_range, sky_spectrum)

print("Calculating Galaxy Shifts and Correcting")
for i, slice in enumerate(galaxy_data):
    interp_slice = splinterp(x_new_range, x_old_range, slice)
    c = cross_corr(interp_slice, interp_gal_spectrum, shifts)
    galaxy_shift = shifts[np.argmax(c)]
    diff = int((avg_shift - galaxy_shift)/10-8)
    interp_slice -= np.roll(interp_sky_spec, diff)
    galaxy_data[i] = interp_slice[::10]

# Cleaning up some bad pixels
for i, rows in enumerate(galaxy_data):
    for j, pixels in enumerate(rows):
        if galaxy_data[i, j] > 200:
            galaxy_data[i, j] = np.mean([galaxy_data[i, j+3], galaxy_data[i, j-3], galaxy_data[i-3,j], galaxy_data[i+3,j]])

plt.imsave('galaxy_sub_sky2.png', galaxy_data, cmap='gray')
save_fits('galaxy_sub_sky2.fits',galaxy_data)

# Problem 14
print("\nProblem 14")

galaxy_low_lim = 40
galaxy_up_lim = 210

doppler_shifts = []
shifts = np.arange(-20,20)

print("Calculating Doppler Shifts")
for spectrum in galaxy_data:
    c = cross_corr(gal_spectrum, spectrum, shifts)
    doppler_shifts.append(shifts[np.argmax(c)])

plt.clf()
plt.title("Doppler Shifts")
plt.xlabel("Galactic Radius")
plt.ylabel("Shift")
plt.plot(doppler_shifts)
plt.savefig('doppler_shifts.png')
print("Saved: doppler_shifts.png'")

good_shifts = doppler_shifts[55:200]

print("Trimming for good shifts.")
plt.clf()
plt.title("Doppler Shifts (Trimmed)")
plt.xlabel("Galactic Radius")
plt.ylabel("Shift")
plt.plot([i for i in range(-70, 75)], good_shifts)
plt.savefig('doppler_shifts_trimmed.png')
print("Saved: doppler_shifts_trimmed.png'")

# Problem 15
print("\nProblem 15")

print("Correcting for intrinsic shift.")
good_shifts = np.array(good_shifts) - np.array(galaxy_shifts[55:200])

plt.plot([i for i in range(-70, 75)], good_shifts)
plt.title("Doppler Shifts (Trimmed + Intrinsic Shifts)")
plt.savefig('doppler_shifts_trimmed_shifted.png')
print("Saved: doppler_shifts_trimmed_shifted.png")

# Problem 16
print("\nProblem 16")
print("See line atlas / spectrum image 'spectrum_overlay.png")

# Problem 17
print("\nProblem 17")
line_1_start = 36
line_1_end = 47
line_2_start = 1013
line_2_end = 1024

line_1 = lamp_spectrum[line_1_start:line_1_end]
line_2 = lamp_spectrum[line_2_start:line_2_end]

print("Fitting Gaussians..")
line_1_center = fitgaussian(line_1)[1][0] + line_1_start
line_2_center = fitgaussian(line_2)[1][0] + line_2_start

print("Line 1 Pixel Center:", line_1_center)
print("Line 2 Pixel Center:", line_2_center)

line_1_wavelength = 6532.8962
line_2_wavelength = 7173.9104

dispersion = (line_2_wavelength - line_1_wavelength) / (line_2_center - line_1_center)

print("Line Dispersion (Angstrom / pixel):", dispersion)

# Problem 18
print("\nProblem 18")
good_shift_center = 70
print("Shift at galaxy center:", good_shifts[good_shift_center])
print("Calculating wavelength shifts")

galaxy_center_pixel = 423.5
galaxy_center_wavelength = line_1_wavelength + wavelength_shift(galaxy_center_pixel-line_1_center, line_1_center, line_1_wavelength, dispersion)

wavelength_shifts = np.zeros(len(good_shifts))

for i, shift in enumerate(good_shifts):
    wavelength_shifts[i] = wavelength_shift(galaxy_center_pixel-shift, galaxy_center_pixel, galaxy_center_wavelength, dispersion)

plt.clf()
plt.title("Doppler Wavelength Shifts")
plt.xlabel("Galactic Radius")
plt.ylabel("Wavelength Shift")
plt.plot([i for i in range(-70, 75)],wavelength_shifts)
plt.savefig('wavelength_shifts.png')
print("Calculating velocities")
velocities = np.zeros(len(good_shifts))

for i, shift in enumerate(wavelength_shifts):
    velocities[i] = ((shift + galaxy_center_wavelength) / galaxy_center_wavelength) * 3E8

plt.clf()
plt.title("Galaxy Spectrum Velocities")
plt.xlabel("Galactic Radius")
plt.ylabel("Velocity")
plt.plot([i for i in range(-70, 75)], velocities)
plt.savefig('galaxy_velocities.png')

# Problem 19
print("\nProblem 19")


# Problem 20
print("\nProblem 20")
redshift = 10100 #km/s
hubble_const = 67 #km/s/Mpc
galaxy_dist = (1 / (hubble_const / redshift))
print("Distance to galaxy:", galaxy_dist, "Mpc.")
print("This seems like a reasonable distance to a low-redshift galaxy however, the latest Simbad measurement places the "
      "distance closer to 194 Mpc.")

# Problem 21
print("\nProblem 21")
galaxy_header = fits.getheader(data_dir+galaxy_files[0])
ccd_scale = fits.getval(data_dir+galaxy_files[0], 'CCDSCALE')
arc_per_degree = 206265

shift_radii = []

for shift in range(-70, 75):
    radii = galaxy_dist * 1000 * shift * ccd_scale / arc_per_degree
    shift_radii.append(radii)

print(shift_radii[-1], 'kpc is an acceptable radius for a spiral galaxy, as they fall '
                       'between the ranges of 5 and 100 kpcs across.')

plt.clf()
plt.title("Galaxy Radial Distances")
plt.xlabel("Pixel Shift")
plt.ylabel("Distance (kpc)")
plt.plot([i for i in range(-70,75)], shift_radii)
plt.savefig('radial_distances.png')

# Problem 22
print("\nProblem 22")
grav_const = 6.67E-11
mass_enclosed = []

for i in range(len(shift_radii)):
    mass = velocities[i]**2 * shift_radii[i] * 3.086E+19 / grav_const
    mass_enclosed.append(mass)

plt.clf()
plt.title("Galaxy Mass Function")
plt.xlabel("Galactic Radius")
plt.ylabel("Mass Enclosed (kg)")
plt.plot(shift_radii, mass_enclosed)
plt.savefig('mass_enclosed.png')

# Problem 24
print("\n Problem 24")
print("Looking at the radial velocity plots, it is clear the velocities of the galaxy arms are not following classic \nKepler-ian "
      "orbital dynamics. Traditionally, one would expect to see the velocity to drop away with increasing distance from \nthe"
      " orbital center, however, according to our data the velocity remains constant after some distance. Additionally, \n"
      "when looking at our calculated mass therefrom, we don't see a falling mass as we'd expect, but rather the mass staying \n"
      "constant yet again. Based on the visible light from the galaxy we can see in our image, the visible mass drops \naway like "
      "we'd expect it to. The conclusion is that there is some non-luminous matter that exists in the bands of the galaxy, \n"
      "increasing its mass and therefore allowing the outer arms to maintain a higher radial velocity.")



