import numpy as np
from numpy import random
from scipy.io import loadmat
import aotools
from scipy import interpolate
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import os
from units_constants import u
from imagesc import imagesc

def DM_PTT_errors(IFdata, gmtpupil_npz, aber_zern_amps):
    """
    Adaptation of Brian McLeod's code takes in DM 292 IF datafile and GMT on 
    axis mask data file, applies masks to IF for DM and PTT and calculates 
    errors and for the full pupil.
    

    Parameters
    ----------
    IFdata : data file
        DM 292 influence matrix, mask.
    gmtpupil_npz : data file
        GMT mask, full mask and individual segments.
    aber_zern_amps : list, floats
        List of Zernike term amp coeffs to include in fit for each GMT segment.
        Size = m x 7, where m is the number of zernike terms.

    Returns
    -------
    DMpoke_full : array_like 
        DM actuator pokes values.
    PTTpoke_full : array_like
        PTT 'actuator' poke values.
    wf_in_full : array_like
        Input wavefront error built from input Zernike mode amplitude coefficients.
    PTTshape_full : array_like
        PTT surface shape.
    DMshape_full : array_like
        DM surface shape.
    wavefront_resid_full : array_like
        Residuals of fit from input wavefront built from Zernike modes.
    PTTDMshape_full : array_like
        PTT+DM shapes combined.
    fitting_error_full : array_like
        Fitting error (residual rms / wf_in rms).

    """

    # DM specifications and calibration data 
    
    # These influence functions are for a mirror with the large-stroke option
    # Ours will be normal-stroke so divide by 1.5 as per email from ALPAO
    IFraw = IFdata['influenceMatrix'] / 1.5 # assume this is surface in microns -- verify later
    nr, nc = IFraw.shape
    IFcalVolts = 10.0 * u.Volt * np.ones((1,nr)) # this is a placeholder for the voltage data from the IF calibration
    calUnits = 1.0 * u.um 
    IF = IFraw * calUnits  # now the influence funtions are defined in meters
    dmsize   = 26.5  # Diameter of DM in mm    
    
    # Only points inside the clear aperture are in the influenceMatrix.
    # The mask is a square grid telling us which of those points are represented.
    IFmask = IFdata['mask'].T.astype(np.bool)  # Empirically determined that we need transpose
    nactuators = IF.shape[0]
    IFcube = np.zeros((IF.shape[0],IFmask.shape[0],IFmask.shape[1]))

    # Copy the influence function points onto the square grid
    for i in range(len(IFcube)):
        IFcube[i][IFmask] = IF[i]

    # Experimental setup

    # GMT pupil mask
    gmtpupil = gmtpupil_npz.f.GMTmask
    beamsize = 24.5  # Diameter of collimated beam at DM in mm

    incident_angle = 20.875

    # The GMT pupil definition is higher resolution than we need.  Downsample it.
    npix  = len(IFmask)
    x = np.arange(npix)
    y = np.arange(npix)
    subsamp = 10
    npixinterp = len(gmtpupil) // subsamp
    gmtsubsamp = gmtpupil[::subsamp,::subsamp]
    gmtmask = gmtsubsamp==1
    IFmasked = np.zeros((nactuators+21,gmtsubsamp.sum().astype(int)))

    # The DM influence functions are defined on a 128x128 grid but less than that is actually used.
    # Figure out what part of the grid is used.

    mask_xmin = IFmask.sum(axis=0).nonzero()[0][0]
    mask_xmax = IFmask.sum(axis=0).nonzero()[0][-1]
    mask_ymin = IFmask.sum(axis=1).nonzero()[0][0]
    mask_ymax = IFmask.sum(axis=1).nonzero()[0][-1]

    # We use a little less than the full diameter of the DM: scale by the ratio of the beam size to the DM size
    #   and account for the stretching of the footprint due to the non-normal incident angle
    xmin = (mask_xmin + mask_xmax) / 2 - (mask_xmax - mask_xmin) / 2 * beamsize / dmsize / np.cos(np.radians(incident_angle))
    xmax = (mask_xmin + mask_xmax) / 2 + (mask_xmax - mask_xmin) / 2 * beamsize / dmsize / np.cos(np.radians(incident_angle))
    ymin = (mask_ymin + mask_ymax) / 2 - (mask_ymax - mask_ymin) / 2 * beamsize / dmsize
    ymax = (mask_ymin + mask_ymax) / 2 + (mask_ymax - mask_ymin) / 2 * beamsize / dmsize

    # Interpolate the influence functions onto the GMT pupil mask
    xnew = np.arange(xmin,xmax,(xmax-xmin)/npixinterp)
    ynew = np.arange(ymin,ymax,(ymax-ymin)/npixinterp)
    IFinterp = np.zeros((IF.shape[0]+21,npixinterp,npixinterp))
    for i in range(len(IFcube)):
        f = interpolate.interp2d(x,y,IFcube[i],kind='linear')
        IFinterp[i] = f(xnew,ynew)
        IFmasked[i] = IFinterp[i][gmtmask]

    # Add PTT mirror piston/tip/tilt to the influence functions
    pttrange = 10  #order of magnitude
    iiy,iix = np.indices((npixinterp,npixinterp))
    ix = iix / (npixinterp / 3)  * pttrange
    iy = iiy / (npixinterp / 3)  * pttrange

    ones = np.ones((npixinterp,npixinterp))

    segcenters_x = np.zeros((7))
    segcenters_y = np.zeros((7))
    segmasks = []

    for i,segmask in enumerate(gmtpupil_npz.f.SegMask):
        segmask = segmask.reshape(gmtpupil.shape)[::subsamp,::subsamp]
        full = np.zeros((npixinterp,npixinterp))
        full[segmask] = ones[segmask]
        IFmasked[nactuators+i*3 + 0] = full[gmtmask]
        IFinterp[nactuators+i*3 + 0] = full

        segcenters_x[i] = ix[segmask].mean()
        full[segmask] = ix[segmask] - ix[segmask].mean()
        IFmasked[nactuators+i*3 + 1] = full[gmtmask]
        IFinterp[nactuators+i*3 + 1] = full

        full[segmask] = iy[segmask] - iy[segmask].mean()
        IFmasked[nactuators+i*3 + 2] = full[gmtmask]
        IFinterp[nactuators+i*3 + 2] = full
        segcenters_x[i] = iix[segmask].mean()
        segcenters_y[i] = iiy[segmask].mean()

        print(iix[segmask].mean(),iiy[segmask].mean() )
        segmasks.append(segmask)

    segmasks = np.array(segmasks)

    # Put zernikes on segments
    
    segradpix = int(npixinterp/6) + 1
    
 
    for which_segment in range(7):
 
        fittingerrors = []
 
        seg = aotools.functions.pupil.circle(segradpix, npixinterp, circle_centre=(segcenters_x[which_segment],segcenters_y[which_segment]))
        
        segmask = segmasks[which_segment]

        # Generate a wavefront with the Zernike terms on a single segment
        zernAmpsList = aber_zern_amps[which_segment,:].tolist()
        segWFE = aotools.functions.zernike.phaseFromZernikes(zernAmpsList,segradpix*2) 
        # segWFE = aotools.functions.zernike.phaseFromZernikes([1],segradpix*2) 
         
        xmin = int(segcenters_x[which_segment]) - segradpix
        xmax = xmin + segradpix * 2
        ymin = int(segcenters_y[which_segment]) - segradpix
        if ymin<0:
            ymin = 0
        ymax = ymin + segradpix * 2
        
        # Paste that into the larger pupil
        seg[ymin:ymax,xmin:xmax] = segWFE
        seg[~segmask] = 0

        # Pick out only the illuminated pixels
        seg_in = seg[gmtmask]
        if which_segment == 0:
            seg_full = np.zeros((7,seg.shape[0],seg.shape[1]))
        
        seg_full[which_segment] = seg

        
 
    bounds = (-1,1)  # Force the DM and PTT to stay in range
    # Perform fit for all segments simultaneously
    
    wf_in_full = seg_full[0]
    for i in [1,2,3,4,5,6]:
        wf_in_full = wf_in_full + seg_full[i]
    wf_to_fit = wf_in_full[gmtmask]
    
    for i in range(10):
            try:
                fit_full = scipy.optimize.lsq_linear(IFmasked.T,wf_to_fit,bounds=bounds)['x']
                break
            except:
                continue
    
    DMpoke_full = fit_full[:292]
    PTTpoke_full = fit_full[292:]
    
    DMshape_full = np.zeros((npixinterp,npixinterp))
    DMshape_full[gmtmask] = fit_full[:292].dot(IFmasked[:292])
    PTTshape_full = np.zeros((npixinterp,npixinterp))
    PTTshape_full[gmtmask] = fit_full[292:].dot(IFmasked[292:])
    PTTDMshape_full = PTTshape_full + DMshape_full
    
    wavefront_resid_full = (wf_in_full - DMshape_full - PTTshape_full) * gmtmask
    pvwavefront_full = np.amax(wavefront_resid_full)-np.amin(wavefront_resid_full)
    pvresid_full = np.amax(wavefront_resid_full)-np.amin(wavefront_resid_full)
    
    fullrms = wf_in_full[gmtmask].std()
    fullresid_rms = wavefront_resid_full[gmtmask].std()
    fitting_error_full = fullresid_rms / fullrms
    
    return DMpoke_full, PTTpoke_full, wf_in_full, PTTshape_full, DMshape_full, wavefront_resid_full, PTTDMshape_full, fitting_error_full, IFcube, IFmasked, IFmask


# Script to test DM_PTT_errors
# Get current directory
here = os.path.abspath(os.path.dirname(__file__))

# Read the influence functions

DM292_path = Path(here, 'BAX377-IF.mat')
IFdata = loadmat(DM292_path)

# Read the GMT pupil mask

gmt_data_path = Path(here, 'GMTmask_onaxis.npz')
gmtpupil_npz = np.load(gmt_data_path)

# number of Zernike terms
# assume input is actual zernike amplitudes, Noll ordered, in units of meters
maxPV = .3 * u.um
m = 12 # number of Zernike modes, starting from piston, per segment
nfreqPerOrder = [1,2,3,4,5]

relamps = np.divide(1,10**np.array(0.01*np.arange(5)))

looplist = [[relamps[i]] * val for i,val in enumerate(nfreqPerOrder)]

llist = []
for i,val in enumerate(nfreqPerOrder):
    llist.extend([relamps[i] ]* val)

#zernAmps = np.tile( np.concatenate(looplist).ravel(), (7,1) )
zernAmps = np.tile( llist, (7,1) )

segErrs = maxPV * zernAmps * np.random.randn(7,sum(nfreqPerOrder))
#segErrslist = [[1],[0],[-1],[0],[0],[0],[0]]
#segErrs = np.array(segErrslist)

DMpokes, PTTpokes, wfs, ptts, dms, resids, pttdm, fitting_error , IFcube, IFmasked, IFmask = DM_PTT_errors(IFdata, gmtpupil_npz, segErrs)

print('fitting error = ', fitting_error)

plt.figure()
plt.imshow(wfs)
plt.colorbar()
plt.title('wfs in')

plt.figure()
plt.imshow(ptts)
plt.colorbar()
plt.title('PTT')

plt.figure()
plt.imshow(dms)
plt.colorbar()
plt.title('DM')

plt.figure()
plt.imshow(pttdm)
plt.colorbar()
plt.title('PTT+DM')

plt.figure()
plt.imshow(resids)
plt.colorbar()
plt.title('residuals')
