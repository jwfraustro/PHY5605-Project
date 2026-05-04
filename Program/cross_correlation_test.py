import sys
sys.path.insert(0, '/mnt/user-data/uploads')
sys.path.insert(0, '.')

import numpy as np
from cross_correlation import cross_correlate_subpixel
from gaussian import gaussian

np.random.seed(42)

x = np.arange(200, dtype=float)
reference = gaussian(x, width=15.0, center=100.0, height=500.0) + 50.0

tests = [
    ('Sub-pixel (2.7 px)',          2.7),
    ('Small sub-pixel (-0.35 px)', -0.35),
    ('Zero shift',                  0.0),
    ('Integer (5.0 px)',            5.0),
    ('Negative (-3.14 px)',        -3.14),
    ('Tiny (0.1 px)',               0.1),
]

print(f"{'Test':30s}  {'True':>8s}  {'Meas':>8s}  {'Resid':>10s}  {'Unc':>10s}")
print('-' * 75)
for name, true_shift in tests:
    shifted = gaussian(x, width=15.0, center=100.0 + true_shift, height=500.0) + 50.0
    shift, err, _, _ = cross_correlate_subpixel(
        shifted, reference, shift_range=(-10, 10), supersample=10
    )
    print(f'{name:30s}  {true_shift:+8.4f}  {shift:+8.4f}  {abs(shift-true_shift):10.6f}  {err:10.6f}')

# Also test with noisy multi-line spectrum
print()
print('Multi-line noisy spectrum tests:')
print('-' * 75)
np.random.seed(42)
x2 = np.arange(1024, dtype=float)
ref2 = np.ones(1024) * 50.0
for c, s in [(100, 300), (350, 200), (520, 500), (700, 150), (900, 250)]:
    ref2 += gaussian(x2, width=3.0, center=float(c), height=s)

for true_shift in [0.5, -1.7, 3.33]:
    shifted2 = np.ones(1024) * 50.0
    for c, s in [(100, 300), (350, 200), (520, 500), (700, 150), (900, 250)]:
        shifted2 += gaussian(x2, width=3.0, center=float(c) + true_shift, height=s)
    shifted2 += np.random.normal(0, 3, 1024)  # add noise

    shift2, err2, _, _ = cross_correlate_subpixel(
        shifted2, ref2, shift_range=(-10, 10), supersample=10
    )
    print(f'  shift={true_shift:+6.2f}  meas={shift2:+8.4f}  resid={abs(shift2-true_shift):10.6f}  unc={err2:10.6f}')