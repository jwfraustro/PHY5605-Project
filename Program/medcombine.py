'''
Routines to do median combination on astronomical data:
  normmedcomb()
  skycormednorm()
'''

import numpy as np

def normmedcomb(data, region=((None, None), (None, None)), finalcorr=False):
  """
    This function does a normalized median combination of the data.

    The routine first divides each sheet of the data by the median
    of the region [y1:y2, x1:x2] in that sheet.  Then it
    median-combines the result.

    Parameters
    ----------
    data: ndarray, 3D
	    Data to combine
    region: tuple of 2 tuples of 2 ints, ((y1, x1), (y2, x2))
	    y1: bottom     of normalization region (default bottom edge)
        x1: left edge  of normalization region (default left   edge)
        y2: top        of normalization region (default top    edge)
        x2: right edge of normalization region (default right  edge)

    Returns
    -------
    normdata: ndarray, 2D
        The normalized, median-combined, 2D data array.
    normfact: ndarray, 1D
      	The normalization factors for each 2D input "sheet".

    Examples
    --------
    >>> import medcombine as mc
    >>> unit = np.ones((5, 5))
    >>> x = np.array([ 5. * unit,
                       7. * unit,
                       5. * unit,])
    >>> mc.normmedcomb(x)
    (array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]]), array([ 5.,  7.,  5.]))

    Revisions
    ---------
    2003-02-26 0.1 jh@oobleck.astro.cornell.edu Initial version.
    2004-03-18 0.2 jh@oobleck.astro.cornell.edu Removed some FITS header
			stuff for easier 2004 class assignment, edited
                        header.  Added sky subtraction (was separate
                        in 2003 problem).
    2007-10-21 0.3 jh@physics.ucf.edu Converted to Python from IDL.
                        Removed header and sky subtraction.
    2008-10-27 0.4 kstevenson@physics.ucf.edu	Updated docstring.
    2009-10-01 0.5 jh@physics.ucf.edu  Tweaked docstring, fixed names, median.
    2009-11-12 0.6 jh@physics.ucf.edu  Tweaked docstring.  Added final corr.
    2016-11-12-0.6 jh@physics.ucf.edu  Merged 2 versions, final corr. optional.
    """
  ((y1, x1), (y2, x2)) = region

  # get sizes, allocate memory
  nz       = data.shape[0]
  normfact = np.zeros(nz)
  normdata = data.copy()

  # normalize each frame
  for k in np.arange(nz):
    normfact[k]  = np.median( normdata[k, y1:y2, x1:x2] )
    normdata[k] /= normfact[k] # YES!  You can omit trailing dimensions
                               # from [k, :, :], so long as you are taking
                               # all elements in them!

  # median combine them
  medcombdat = np.median(normdata, axis=0)

  # Correct for the fact that 1.0 needn't actually appear in the frame
  # (this is a tiny correction).  Suggested by P. Cubillos, 2009.
  if finalcorr:
    medcorr = np.median(medcombdat)
    medcombdat /= medcorr
    normfact   *= medcorr

  return medcombdat, normfact

def skycormednorm(objdata, normskydata, region=((None, None), (None, None))):
  """
    Denormalize the sky frame and remove it from the input data.

    This is the second half of normalized median combination.
    The routine finds the normalization of the object frame by
    taking the median of objdata[y1:y2, x1:x2].  It multiplies this by
    normskydata.  It subtracts the result from objdata.

    Parameters
    ----------
    objdata : ndarray, 2D
        Object frame to correct
    normskydata : ndarray, 2D
        Normalized sky frame
    region: tuple of 2 tuples of 2 ints, ((y1, x1), (y2, x2))
	    y1: bottom     of normalization region (default bottom edge)
        x1: left edge  of normalization region (default left   edge)
        y2: top        of normalization region (default top    edge)
        x2: right edge of normalization region (default right  edge)

    Returns
    -------
    output: ndarray, 2D
        The corrected object frame.

    Examples
    --------
    None yet.

    Revisions
    ---------
    2003-02-26 0.1 jh@oobleck.astro.cornell.edu Initial version.
    2004-04-03 0.2 jh@oobleck.astro.cornell.edu Modified for 2004 assignment.
    2004-04-05 0.3 jh@oobleck.astro.cornell.edu Fixed to use x1, x2, y1,
                  y2, and to subtract dark before sky normalization
                  calc.
    2007-10-21 0.4 jh@physics.ucf.edu Converted from IDL to Python.
                  Remove header stuff.
    2008-10-27 0.5 kstevenson@physics.ucf.edu  Updated docstring
    2009-10-01 0.6 jh@physics.ucf.edu  Tweaked docstring.  Shortened program.
    2009-11-12 0.7 jh@physics.ucf.edu  Tweaked docstring.
    2016-11-12 0.8 jh@physics.ucf.edu  Tweaked docstring.
  """
  ((y1, x1), (y2, x2)) = region            # set corners
  retval = objdata.copy()                  # copy the data
  norm   = np.median(objdata[y1:y2,x1:x2]) # calculate the normalization
  retval = retval - norm * normskydata     # de-normalize sky and subtract
  
  return retval
