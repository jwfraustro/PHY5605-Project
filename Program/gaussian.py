'''
    Name
    ----
    gaussian

    File
    ----
    gaussian.py

    Description
    -----------
    Routines for evaluating, estimating parameters of, and fitting Gaussians.

    Package Contents
    ----------------
    N-dimensional functions:

    gaussian(x, width=1., center=0., height=None, params=None)
        Evaluate the Gaussian function with given parameters at x
        (n-dimensional).
    fitgaussian(y, x)
        Calculates a Gaussian fit to (y, x) data, returns (width,
        center, height).

    1-dimensional functions:
    
    gaussianguess(y, x=None)
        Crudely estimates the parameters of a Gaussian that fits the
        (y, x) data.

    Examples:
    ---------
    See fitgaussian() example.

    Revisions
    ---------
    2007-09-17 0.1 jh@physics.ucf.edu Initial version 0.01, portions
                   adapted from http://www.scipy.org/Cookbook/FittingData.
    2007-10-02 0.2 jh@physics.ucf.edu Started making N-dimensional,
                   put width before center in args.
    2007-11-13 0.3 jh@physics.ucf.edu Made N-dimensional.
    2008-12-02 0.4 nlust@physics.ucf.edu Made fit gaussian return errors, and
                   fixed a bug generating initial guesses
    2009-10-25 0.5 jh@physics.ucf.edu Standardized all headers, fixed
                   an error in a fitgaussian example, added example
                   ">>>"s and plot labels.
    2009-11-14 0.6 jh@physics.ucf.edu Fixed gaussianguess() to work in
                   N dimensions, fixed docs and examples of all
                   functions, made everything consistent with x as
                   indices or linear in 1D case.
    2010-12-15 0.7 jh@physics.ucf.edu Tested for success in fitgaussian.
    2011-10-11 0.8 jh@physics.ucf.edu Fixed those fitgaussian test,
                   added access to minimizer input and output.
    2014-10-01 0.9 jh@physics.ucf.edu Converted to Python 3.
    2014-11-24 0.10 jh@physics.ucf.edu Ndim bug fixed, see below.
    2016-10-25 0.11 jh@physics.ucf.edu Changed == to "is" and != to "is not"
                    in None comparisons.
    2017-10-10 0.12 jh@physics.ucf.edu Integer division of pdim to use as
                    index in Python3.
'''

import numpy as np
import scipy.optimize as so

def gaussian(x, width=1., center=0., height=None, param=None):
  '''
    Evaluates the Gaussian with given parameters at locations in x.

    Parameters
    ----------
    x : ndarray
        Abcissa values.  Arranged as the output of np.indices() but
        may be float.  The highest dimension must be equal to the
        number of other dimensions (i.e., if x has 6 dimensions, the
        highest dimension must have length 5, and x[i] must give the
        coordinates on axis i).  May also just give 1D abcissa values
        as a 1D array.

    width : array_like
        The width of the Gaussian function, sometimes called sigma.
        If scalar, assumed constant for all dimensions.  If array-like,
        must be linear and the same length as the first dimension of
        x.  In this case, each element gives the width of the function
        in the corresponding dimension.  Default: (1.,).

    center : array_like
        The mean location of the Gaussian function, sometimes called
        x0.  Same format and behavior as width.  Default: (0.,).

    height : scalar
        The height of the Gaussian at its center.  If not set,
        initialized to the value that makes the Gaussian integrate to
        1.  If you want it to integrate to another number, leave
        height alone and multiply the result by that other number
        instead.  Must be scalar.  Default: [np.product(1./sqrt(2 * pi
        * width**2))].

    param : ndarray or tuple, 3-element
        Instead of giving width, center, and height separately, give
        them in an array-like object, concatenated in this order.  So,
        if width=(2,3), center=(5,6), height=9, using
        param=[2,3,5,6,9] gives the same effect.  If param is defined,
        width, center, and height are ignored and may be overwritten.
        This is useful in fitting functions.

    Returns
    -------
    results : ndarray, same shape as x (or first element of x if
        multidimensional) 
        This function returns the Gaussian function of the given
        width(s), center(s), and height applied to its input.  The
        Gaussian function is: f(x) = 1./sqrt(2 * pi * width**2) *
        exp(-0.5 * ((x - center) / width)**2).  It is defined in
        multiple dimensions as the product of orthogonal,
        single-dimension Gaussians.

    Examples
    --------
    
    >>> import matplotlib.pyplot as plt
    >>> import gaussian as g
    
    >>> x = np.linspace(-10., 10., 2001)
    >>> plt.plot(x, g.gaussian(x))
    >>> plt.title('Gaussian')
    >>> plt.xlabel('Abcissa')
    >>> plt.ylabel('Ordinate')
    
    >>> # use an array [3] as a single parameter vector
    >>> z = np.array([2., 2, 3])
    >>> plt.plot(x, g.gaussian(x, *z))

    >>> # Test that it integrates to 1.
    >>> a = np.indices([100, 100]) - 50
    >>> print(np.sum(g.gaussian(a, 3, 3)))
    1.0
    >>> print(np.sum(g.gaussian(a, np.array([1,2]), np.array([2,3]))))
    1.00000000535

    >>> plt.clf()
    >>> plt.imshow(g.gaussian(a, [3,5], [7,3]))
    >>> plt.title('2D Gaussian')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')

    >>> plt.clf()
    >>> plt.imshow(g.gaussian(a, param=[3,2,7,3]))
    >>> plt.title('2D Gaussian')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')

    Revisions
    ---------
    2007-09-17 0.1  jh@physics.ucf.edu	Initial version 0.01
    2007-10-02 0.2  jh@physics.ucf.edu	Started making N-dimensional,
    					put width before center in args.
    2007-11-13 0.3  jh@physics.ucf.edu	Fixed docs, bugs, added param,
    					made N-dimensional
    2009-10-01 0.4  jh@physics.ucf.edu	Fixed docs.
    2009-10-25 0.5  jh@physics.ucf.edu	Added examples and plot labels.
    2009-11-14 0.6  jh@physics.ucf.edu	Fixed doc for x, change example.
    2014-10-07 0.7  jh@physics.ucf.edu  Convert to Python 3.
    2016-10-25 0.11 Changed == to "is" and != to "is not" in None comparisons.
 '''
  if param is not None:               # unpack parameters, if necessary
    pdim = len(param)
    if pdim // 2 == pdim / 2.:
      pdim = pdim // 2
    else:
      pdim = (pdim - 1) // 2
      height = param[-1]
    width  = param[     :     pdim]
    center = param[pdim : 2 * pdim]

  ndim = x.ndim - 1
  if ndim == 0:    # 1D case may have shape (n1,) or (1, n1)
    ndim = 1
    oldshape = x.shape
    x.shape = (1, x.shape[0])
  if type(center) != np.ndarray:
    center += np.zeros(ndim)
  if type(width) != np.ndarray:
    width  += np.zeros(ndim)
  r2pi = np.sqrt(2. * np.pi)
  if height is None:
    height = np.product(1. / (width * r2pi))
  ponent = 0.
  for i in np.arange(ndim):
    ponent += ( (x[i] - center[i]) / width[i] )**2
  if 'oldshape' in locals():
    x.shape = oldshape
  return height * np.exp(-0.5 * ponent)

def gaussianguess(y, x=None):
  '''
    Crudely estimates the parameters of a Gaussian that fits the (y, x) data.

    Parameters
    ----------
    y : ndarray
        The function values.
    x : ndarray, same shape as np.indices(y)
        (optional) An array giving the abcissas of y in the format of
        np.indices().  Must be sorted ascending (which is not
        checked).  Default: np.indices(y)

    Returns
    -------
    param : tuple, 3 elements
    This function returns a tuple giving extimates of the (width,
    center, height) of a Gaussian that might fit the input data.
    See 'param' input parameter of gaussian() for format of this
    tuple.

    Notes
    -----
    If the data do not look Gaussian, and certainly if they contain
    spikes higher/lower than the peak of the real Gaussian, the
    parameter estimates will be poor.  x must be sorted ascending
    (which is not checked).  Irregular grids might give odd results.

    Method: The most extreme element of y is the location and height
    of the peak.  The routine looks for the width at 0.6 of this
    maximum by sampling along the linear slice in each dimension
    through the peak location.  If only one pixel is above 0.6 of the
    peak, the width is returned as 1.  This prevents fitgaussian from
    blowing up.  In this case it's difficult to establish a real
    width, anyway.

    Examples
    --------
    >>> import gaussian as g

    >>> # 1D example
    >>> x = np.linspace(-10., 10., 201)
    >>> y = g.gaussian(x)
    >>> print(g.gaussianguess(y, x))
    ((1.0,), (0.0,), 0.3989422804014327)
    
    >>> # 2D example
    >>> x = np.indices([100, 100]) - 50
    >>> y = g.gaussian(x, (3,5), (13,14))
    >>> print(g.gaussianguess(y, x))
    ((3.0, 5.0), (13, 14), 0.010610329539459692)

    Revisions
    ---------
    2007-09-17 0.1 jh@physics.ucf.edu    Initial version 0.01
    2007-11-13 0.2 jh@physics.ucf.edu    Fixed docs, return order.
    2008-12-02 0.3 nlust@physics.ucf.edu Fixed a bug where if an
                   initial guess was not provided, it would error out
    2009-10-25 0.4 jh@physics.ucf.edu    Converted to standard doc header.
    2009-11-14 0.5 jh@physics.ucf.edu    Made it work in N dimensions,
                   add examples, docs.
    2014-10-07 0.6 jh@physics.ucf.edu    Convert to Python 3, update examples.
    2014-11-24 0.7 jh@physics.ucf.edu    Add ndim line to x==None branch.
                   make sure widths are never 0.
    2016-10-25 0.11 Changed == to "is" and != to "is not" in None comparisons.
    '''
  if x is None :
    x = np.indices(y.shape)
    ndim = x.ndim - 1
  else:
    ndim = x.ndim - 1
    if ndim == 0:    # 1D case may have shape (n1,) or (1, n1)
      ndim = 1
      oldshape = x.shape
      x.shape = (1, x.shape[0])
    elif x.shape != tuple(np.concatenate(((ndim,), y.shape))) :
      raise ArrayShapeError("x must have compatible shape with y (and be sorted).")

  iymax  = y.argmax()
  ymax   = y.flat[iymax]
  
  iymin  = y.argmin()
  ymin   = y.flat[iymin]
  
  if np.abs(ymin) >= np.abs(ymax) :
    icenter = iymin
  else :
    icenter = iymax
  
  icenter = np.unravel_index(icenter, y.shape)
  height = y[icenter]
  
  # To guess the width in each direction, we extract the 1D slice in
  # that dimension that goes through center...
  width = np.zeros(ndim)
  allslice = slice(0, -1)
  for i in np.arange(ndim):
    theslice = list(icenter)
    theslice[i] = allslice
    sl = y[tuple(theslice)]
    # then find where in that slice it drops below 0.6 times the
    # height in both directions...
    gtsigma = (np.where(sl > (0.6 * height)))[0]
    imax = gtsigma.max()
    imin = gtsigma.min()
    # find the coordinates of those points in the input x...
    ixmax = list(icenter)
    ixmin = list(icenter)
    ixmax[i] = imax
    ixmin[i] = imin
    # then take half that interval.
    width[i] = (  x[(i,) + tuple(ixmax)]
                - x[(i,) + tuple(ixmin)] ) / 2.
    if width[i] == 0:      # otherwise fitgaussian will blow up
      width[i] = 1.

  # find center in x coordinates (not index coords)  
  center = list(icenter)
  for i in np.arange(ndim):
    center[i] = x[(i,) + icenter]

  if 'oldshape' in locals():
    x.shape = oldshape
  
  return (tuple(width), tuple(center), height)

def fitgaussian(y, x=None, guess=None, lsargs={"maxfev":2000}, minout=None):
  '''
    Fits an N-dimensional Gaussian to (value, coordinate) data.

    Parameters
    ----------
    y : array_like
        Array giving the values of the function.
    x : ndarray, same shape as np.indices(y)
        (optional) An array giving the abcissas of y in the format of
        np.indices().  Must be sorted ascending (which is not checked)
        if guess is not given.  Default: np.indices(y)
    guess : tuple, (width, center, height)
        Tuple giving an initial guess of the Gaussian parameters for
        the optimizer.  If supplied, x and y can be any shape and need
        not be sorted.  See gaussian() for meaning and format of this
        tuple.
    lsargs : dictionary
        Optional arguments to scipy.optimize.leastsq().  Interesting
        ones include maxfev, ftol, xtol, and gtol.  Do not redefine
        full_output, as the code depends on the value it sets for this
        parameter.  By default, maxfev is set to 2000 since the
        default in the minimizer is too low.
    minout : Boolean
        If true, append the minimizer's output tuple to the other outputs.

    Returns
    -------
    width : ndarray, one element per dimension
        The fitted Gaussian widths in each dimension.
    center : ndarray, one element per dimension
        The fitted Gaussian center coordinate in each dimension.
    height : ndarray scalar
        The fitted height.
    err : ndarray, 2 * number-of-dims + 1 elements
        An array containing the concatenated uncertainties for width,
        center, and height, ordered as above.  For example, 2D input gives
        np.array([widthyerr, widthxerr, centeryerr, centerxerr, heighterr]).
    lsout : tuple
        (Optional) Minimizer output, all in one tuple.  Included only
        if minout=True.  See scipy.optimize.leastsq() for
        documentation of its contents.

    Notes
    -----
    If the input does not look anything like a Gaussian, the result
    might not even be the best fit to that.

    Method: First guess the parameters (if no guess is provided), then
    call a Levenberg-Marquardt optimizer to finish the job.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import gaussian as g

    >>> # parameters for X
    >>> lx = -3.  # low end of range
    >>> hx = 5.   # high end of range
    >>> dx = 0.05 # step

    >>> # parameters of the noise
    >>> nc = 0.0  # noice center
    >>> ns = 1.0  # noise width
    >>> na = 0.2  # noise amplitude

    >>> # 1D Example

    >>> # parameters of the underlying Gaussian
    >>> wd = 1.1  # width
    >>> ct = 1.2  # center
    >>> ht = 2.2  # height

    >>> # x and y data to fit
    >>> x  = np.arange(lx, hx + dx / 2., dx)
    >>> x +=                             na * np.random.normal(nc, ns, x.size)
    >>> y  = g.gaussian(x, wd, ct, ht) + na * np.random.normal(nc, ns, x.size)
    >>> s  = x.argsort()    # sort, in case noise violated order
    >>> xs = x[s]
    >>> ys = y[s]

    >>> # calculate guess and fit
    >>> (width, center, height)     = g.gaussianguess(ys, xs)
    >>> (fw,    fc,     fh,    err) = g.fitgaussian(ys, xs)

    >>> # plot results
    >>> plt.clf()
    >>> plt.plot(xs, ys)
    >>> plt.plot(xs,      g.gaussian(xs, wd,    ct,     ht))
    >>> plt.plot(xs,      g.gaussian(xs, width, center, height))
    >>> plt.plot(xs,      g.gaussian(xs, fw,    fc,     fh))
    >>> plt.title('Gaussian Data, Guess, and Fit')
    >>> plt.xlabel('Abcissa')
    >>> plt.ylabel('Ordinate')
    >>> # plot residuals
    >>> plt.clf()
    >>> plt.plot(xs, ys - g.gaussian(xs, fw,    fc,     fh))
    >>> plt.title('Gaussian Fit Residuals')
    >>> plt.xlabel('Abcissa')
    >>> plt.ylabel('Ordinate')

    >>> # 2D Example

    >>> # parameters of the underlying Gaussian
    >>> wd = (1.1, 3.2)  # width
    >>> ct = (1.2, 3.1)  # center
    >>> ht = 2.2         # height

    >>> # x and y data to fit
    >>> nx = (hx - lx) / dx + 1
    >>> x  = np.indices((nx, nx)) * dx + lx
    >>> y  = g.gaussian(x, wd, ct, ht) + na * np.random.normal(nc, ns, x.shape[1:])

    >>> # calculate guess and fit
    >>> (width, center, height)     = g.gaussianguess(y, x)
    >>> (fw,    fc,     fh,    err) = g.fitgaussian(y, x)

    >>> # plot results
    >>> plt.clf()
    >>> plt.title('2D Gaussian Given')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')
    >>> plt.imshow(    g.gaussian(x, wd,    ct,     ht))
    >>> plt.clf()
    >>> plt.title('2D Gaussian With Noise')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')
    >>> plt.imshow(y)
    >>> plt.clf()
    >>> plt.title('2D Gaussian Guessed')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')
    >>> plt.imshow(    g.gaussian(x, width, center, height))
    >>> plt.clf()
    >>> plt.title('2D Gaussian Fit')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')
    >>> plt.imshow(    g.gaussian(x, fw,    fc,     fh))
    >>> plt.clf()
    >>> plt.title('2D Gaussian Fit Residuals')
    >>> plt.xlabel('X')
    >>> plt.ylabel('Y')
    >>> plt.imshow(y - g.gaussian(x, fw,    fc,     fh))

    >>> # All cases benefit from...

    >>> # show difference between fit and underlying Gaussian
    >>> # Random data, your answers WILL VARY.
    >>> np.array(fw) - np.array(wd)
    array([-0.00015463,  0.01373218])
    >>> np.array(fc) - np.array(ct)
    array([ 0.00061797, -0.00113882])
    >>> np.array(fh) - np.array(ht)
    -0.0026334217737993271

    Revisions
    ---------
    2007-09-17 0.1 jh@physics.ucf.edu	Initial version 0.01,
                   portions adapted from
                   http://www.scipy.org/Cookbook/FittingData.
    2007-11-13 0.2 jh@physics.ucf.edu    Made N-dimensional.
    2008-12-02 0.3 nlust@physics.ucf.edu Included error
                   calculation, and return Fixed a bug in which if the
                   initial guess was None, and incorrect shape array
                   was generated. This caused gaussian guess to fail.
    2009-10-25 0.4 jh@physics.ucf.edu    Converted to standard doc
                   header, fixed examples to return 4 parameters.
    2009-11-14 0.5 jh@physics.ucf.edu    Fixed examples for
                   N-dimensional gaussianguess(), fixed code spacing.
    2010-12-15 0.6 jh@physics.ucf.edu    Tested for success in leastsq.
    2011-10-11 0.7 jh@physics.ucf.edu    Fixed test for success,
                   added argument passing and full return to/from minimizer.
    2014-10-07 0.8 jh@physics.ucf.edu    Convert to Python 3, update examples.
    2014-11-24 0.9 jh@physics.ucf.edu    Change 2x np.append into 1x np.hstack.
    2016-10-25 0.11 Changed == to "is" and != to "is not" in None comparisons.
    2017-10-10 0.12 Integer division of pdim to use as index in Python3.
  '''
  if x is None :
    x = np.indices(y.shape)
  else:
    if (   ((x.ndim == 1) and (x.shape     != y.shape))
        or ((x.ndim >  1) and (x.shape[1:] != y.shape))):
      raise ValueError("x must give coordinates of points in y.")

  if guess is None:
    guess = gaussianguess(y, x)

  # "ravel" the guess
  gss = np.hstack((guess[0], guess[1], guess[2]))

  # fit
  residuals = lambda p: np.ravel(gaussian(x, param=p) - y)
  lsout = so.leastsq(residuals, gss, full_output=True, **lsargs)
  p, cov, info, mesg, success = lsout
  if   success < 1 \
    or success > 4:
    raise ValueError(mesg + " No convergence.  Guess = "+str(guess) \
      +"; try a different guess, maxfev, or tolerances.\n See scipy.optimize.leastsq() docs.")
  err = np.sqrt(np.diagonal(cov))
  # unravel the result
  pdim = len(p)
  pdim = (pdim - 1) // 2 # integer division to use as an index
  width  = p[     :     pdim]
  center = p[pdim : 2 * pdim]
  height = p[-1]

  ret = (width, center, height, err)

  if minout:
    ret = (width, center, height, err, lsout)

  return ret
