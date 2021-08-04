import numpy as np
from Add_PTT_IFs import Add_PTT_IFs
from fit_DM_PTT import fit_DM_PTT
from prepare_DM292_IFs import prepare_DM292_IFs

def DM_PTT_projector(gmtmask, segmasks, wf_to_fit):
    """
    Adaptation of Brian McLeod's code loads DM292 IF datafile and takes GMT on-axis 
    mask, applies masks to IF for DM and PTT, and calculates required actuator pokes 
    for all the segments.

    Parameters
    ----------
    gmtmask : array_like, boolean
        GMT full mask of size npix x npix. DM IFs will be interpolated to match gmtmask.
    segmasks : array_like, boolean
        GMT segment masks. Three dimensional array (7 x npix x npix)
    wf_to_fit : array_like, float
        Input wf phase error of size npix x npix.

    Returns
    -------
    DMpoke : array_like 
        DM actuator pokes values.
    PTTpoke : array_like
        PTT 'actuator' poke values.
    PTTshape : array_like
        PTT surface shape.
    DMshape : array_like
        DM surface shape.
    wavefront_resid : array_like
        Residuals of fit from input wavefront built from Zernike modes.
    PTTDMshape : array_like
        PTT+DM shapes combined.
    fitting_error : array_like
        Fitting error (residual rms / wf_in rms).

    """
    
    # beamsizePTT = 51.0 * u.mm # Diameter of collimated beam at PTT in mm
    # eventually will use this number to make ruler for input WFPT pupil

    # GMT pupil mask size in pixels
    npixinterp = gmtmask.shape[0]
    
    IFmasked = prepare_DM292_IFs(gmtmask)
    
    # Add PTT mirror piston/tip/tilt to the influence functions
    pttrange = 10  #order of magnitude
    IFmasked = Add_PTT_IFs(IFmasked, gmtmask, segmasks, npixinterp, pttrange)
    
    # Do least-squares fit to the input wf error with influence functions
    fit = fit_DM_PTT(IFmasked, wf_to_fit)
    
    DMpoke = fit[:292]
    PTTpoke = fit[292:]
    
    DMshape = np.zeros((npixinterp,npixinterp))
    DMshape[gmtmask] = fit[:292].dot(IFmasked[:292])
    PTTshape = np.zeros((npixinterp,npixinterp))
    PTTshape[gmtmask] = fit[292:].dot(IFmasked[292:])
    PTTDMshape = PTTshape + DMshape
    
    wavefront_resid = (wf_to_fit - DMshape - PTTshape) * gmtmask
    # pvwavefront_full = np.amax(wavefront_resid)-np.amin(wavefront_resid)
    # pvresid_full = np.amax(wavefront_resid)-np.amin(wavefront_resid)
    
    fullrms = wf_to_fit[gmtmask].std()
    fullresid_rms = wavefront_resid[gmtmask].std()
    fitting_error = fullresid_rms / fullrms
    
    return DMpoke, PTTpoke, PTTshape, DMshape, wavefront_resid, PTTDMshape, fitting_error
