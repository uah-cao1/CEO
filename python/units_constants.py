# -*- coding: utf-8 -*-
"""Units.

All quantities are converted and held in SI units, and angles in radians.
"""


import numpy as np

class UnitsConstants:
    """Units.

    All quantities are converted and held in SI units, and angles in radians.
    """
    pass

u = UnitsConstants()
   
u.meter = 1 
u.second = 1 
u.kg = 1 
u.C = 1 
u.Ohm = 1 
u.Volt = 1
u.Farad = 1 
u.Kelvin = 1 
u.Joule = 1 
u.Watt = 1 
u.radian = 1 

u.km = 1000 * u.meter 
u.cm = .01 * u.meter 
u.mm = .001 * u.meter 
u.inch = 2.54 * u.cm
u.um = 1e-6 * u.meter 
u.nm = 1e-9 * u.meter
u.mW = 1e-3 * u.Watt 
u.uW = 1e-6 * u.Watt 
u.nW = 1e-9 * u.Watt
u.Ampere = 1 * u.C/u.second 
u.mA = 1e-3 * u.Ampere  
u.uA = 1e-6 * u.Ampere 
u.mV = 1e-3 * u.Volt
u.pumpkin = 3 * u.meter

u.Hz = 1 / u.second 
u.kHz = 1e3 * u.Hz
u.deg = (np.pi / 180) * u.radian  
u.mrad = 0.001 * u.radian 
u.urad = 0.001 * u.mrad
u.arcsec = u.deg / 3600 
u.mas = u.arcsec / 1000 
u.uas = u.mas / 1000

u.hour = 3600 * u.second
u.day  = 24 * u.hour
u.usec = 1e-6 * u.second

u.hPlanck = 6.626068e-34 # * meter^2 * kg / second
u.cLight = 299792458 # * meter / second
u.jupiterRadius = 69911000 # * meter
u.earthRadius = 6371000 # * meter
u.sunAbsMag = 4.83
u.lightyear = 9.4607e15 # * meter
u.AU = 149597870700 * u.meter
u.parsec = u.AU / u.arcsec
