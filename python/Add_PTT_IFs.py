import numpy as np

def Add_PTT_IFs(IFmasked, gmtmask, segmasks, npix, pttrange):
    """
    Adaptation of Brian McLeod's code to add PTT influence functions to DM292
    influence functions. Output is an array of masked influence functions.

    Parameters
    ----------
    IFmasked : array_like
        Two dimentional array (313 x length of masked vector). DM292 IFs 
        should have already been interpolated to to same size as GMT mask and 
        correctly aligned.
    gmtmask : array_like, boolean
        GMT mask square array. This is the full working size.
    segmasks : array_like, boolean
        GMT segment mask square array. Three dimansional array (7 x npix x npix)
    npix : int
        Number of pixels across GMT pupil/mask.
    ptt_range : float
        Range of motion of PTT actuators.

    Returns
    -------
    IFmasked_out : array_like
        Same format and size as IFmasked.

    """
    
    # Add PTT mirror piston/tip/tilt to the influence functions
    nactuators = 292
    iiy,iix = np.indices((npix,npix))
    ix = iix / (npix / 3)  * pttrange
    iy = iiy / (npix / 3)  * pttrange

    ones = np.ones((npix,npix))

    for i in range(7):
        segmask = segmasks[i]
        full = np.zeros((npix,npix))
        full[segmask] = ones[segmask]
        IFmasked[nactuators+i*3 + 0] = full[gmtmask]

        full[segmask] = ix[segmask] - ix[segmask].mean()
        IFmasked[nactuators+i*3 + 1] = full[gmtmask]

        full[segmask] = iy[segmask] - iy[segmask].mean()
        IFmasked[nactuators+i*3 + 2] = full[gmtmask]

    IFmasked_out = IFmasked
    return IFmasked_out