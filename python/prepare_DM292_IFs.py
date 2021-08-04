import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
from scipy import interpolate
from units_constants import u

def prepare_DM292_IFs(gmtmask):
    """
    Reads in DM292 .mat file and creates array with raw influence functions. Then, 
    interpolates 

    Parameters
    ----------
    gmtmask : array_like, boolean
        GMT full mask of size npix x npix. DM IFs will be interpolated to match gmtmask.

    Returns
    -------
    IFmasked : array_like, 2d
        DM 292 influence functions (IFs) interpolated to same size as GMT mask 
        and vectorized. Size of output array is 313 x nvect, where nvec is 
        the vector for each IF with valid data according to the mask.

    """
    # Get current directory
    here = os.path.abspath(os.path.dirname(__file__))

    # Read the influence functions
    DM292_path = Path(here, 'BAX377-IF.mat')
    IFdata = loadmat(DM292_path)
    
    # These influence functions are for a mirror with the large-stroke option
    # Ours will be normal-stroke so divide by 1.5 as per email from ALPAO
    IFraw = IFdata['influenceMatrix'] / 1.5 # assume this is surface in microns -- verify later
    nr, nc = IFraw.shape
    
    # DM specifications and calibration data
    IFcalVolts = 10.0 * u.Volt * np.ones((1,nr)) # this is a placeholder for the voltage data from the IF calibration
    calUnits = 1.0 * u.um 
    IF = IFraw * calUnits  # now the influence funtions are defined in meters
    dmsize   = 26.5 * u.mm # Diameter of DM in mm
    beamsizeDM = 24.5 * u.mm # Diameter of collimated beam at DM in mm
    # incident_angle = 20.875 # true value
    incident_angle = 0.0 # for testing
    
    # Only points inside the clear aperture are in the influenceMatrix.
    # The mask is a square grid telling us which of those points are represented.
    IFmask = IFdata['mask'].T.astype(np.bool)  # Empirically determined that we need transpose
    nactuators = IF.shape[0]
    IFcube = np.zeros((IF.shape[0],IFmask.shape[0],IFmask.shape[1]))
    
    # Copy the influence function points onto the square grid
    for i in range(len(IFcube)):
        IFcube[i][IFmask] = IF[i]
    
    # GMT pupil mask size
    npixinterp = gmtmask.shape[0]
    
    npix  = len(IFmask)
    x = np.arange(npix)
    y = np.arange(npix)
    gmtmaskint = gmtmask.astype(int)
    IFmasked = np.zeros((nactuators+21,gmtmaskint.sum().astype(int)))
    
    # The DM influence functions are defined on a 128x128 grid but less than that is actually used.
    # Figure out what part of the grid is used.

    mask_xmin = IFmask.sum(axis=0).nonzero()[0][0]
    mask_xmax = IFmask.sum(axis=0).nonzero()[0][-1]
    mask_ymin = IFmask.sum(axis=1).nonzero()[0][0]
    mask_ymax = IFmask.sum(axis=1).nonzero()[0][-1]

    # We use a little less than the full diameter of the DM: scale by the ratio of the beam size to the DM size
    #   and account for the stretching of the footprint due to the non-normal incident angle
    xmin = (mask_xmin + mask_xmax) / 2 - (mask_xmax - mask_xmin) / 2 * beamsizeDM / dmsize / np.cos(np.radians(incident_angle))
    xmax = (mask_xmin + mask_xmax) / 2 + (mask_xmax - mask_xmin) / 2 * beamsizeDM / dmsize / np.cos(np.radians(incident_angle))
    ymin = (mask_ymin + mask_ymax) / 2 - (mask_ymax - mask_ymin) / 2 * beamsizeDM / dmsize
    ymax = (mask_ymin + mask_ymax) / 2 + (mask_ymax - mask_ymin) / 2 * beamsizeDM / dmsize
    
    # Interpolate the influence functions onto the GMT pupil mask
    xnew = np.arange(xmin,xmax,(xmax-xmin)/npixinterp)
    ynew = np.arange(ymin,ymax,(ymax-ymin)/npixinterp)
    IFinterp = np.zeros((IF.shape[0]+21,npixinterp,npixinterp))
    for i in range(len(IFcube)):
        f = interpolate.interp2d(x,y,IFcube[i],kind='linear')
        IFinterp[i] = f(xnew,ynew)
        IFmasked[i] = IFinterp[i][gmtmask]
    
    return IFmasked