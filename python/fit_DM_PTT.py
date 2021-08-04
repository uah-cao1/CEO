import scipy

def fit_DM_PTT(IFmasked, wf_to_fit):
    """Produce least-squares fit to input wf error map with DM+PTT 
    infleunce functions. Output is fit array

    Parameters
    ----------
    IFmasked : array_like
        Two dimentional array (313 x length of masked vector). DM292 IFs 
        should have already been interpolated to to same size as GMT mask 
        (npix x npix) and correctly aligned.
    wf_to_fit : array_like, float
        Input wf phase error of size npix x npix.

    Returns
    -------
    fit : array_like, float
        1-d array of 313 actuator poke values. First 292 are for the DM.
    """
    
    bounds = (-1,1)  # Force the DM and PTT to stay in range
    
    # Perform fit for all segments simultaneously
    for i in range(10):
            try:
                fit = scipy.optimize.lsq_linear(IFmasked.T,wf_to_fit,bounds=bounds)['x']
                break
            except:
                continue

    return fit