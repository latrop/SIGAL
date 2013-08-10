# Setup file for the program model.py
#!/usr/bin/python
# -*- coding:  cp1251 -*-
#import pics
import pylab
import sys
import os
import math
import numpy as np
from scipy import stats
from scipy import special
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.ticker import NullFormatter
from numpy import *
from pylab import *
from scipy import special
tmp_out = sys.stdout
import random as random_number
import random


# Output files:
par_file = 'models.dat'
file_image = 'galaxy_model.fits'
file_psf = 'psf.fits'



# PSF
box_psf = 50
window = 'gauss'

nx_psf = box_psf
ny_psf = box_psf

# Contaminents
Nmax_stars = 20
Nmax_gals = 1

# Sky
dskyX = 0.
dskyY = 0.

def disk(ima_filter):
	if ima_filter=='r':
		b1 = random_number.normalvariate(19.155,0.514)
		k1 = random_number.normalvariate(2.978,0.0)

	return k1,b1

q0 = 0.18
incl_lim = 70.	# 85.

def bulge(ima_filter,reb):
	if ima_filter=='r':
		if reb<1.3:
			A = 2.57843008584e-07
			B = random_number.normalvariate(-3.04296910626,0.12) #0.05
			C = random_number.normalvariate(0.379396518596,0.12) #0.06
		if reb>=1.3:
			A = 2.57843008584e-07
			B = random_number.normalvariate(-3.04296910626,0.05) #0.05
			C = random_number.normalvariate(0.379396518596,0.06) #0.06
	if reb<=C:
		#print 'reb<c!!!'
		return 0.
	else:

		return	B*log10( (reb-C)/A )


def rel_numb(gal_type,scale):
	if gal_type=='all' and scale<1.:
		disk_frac = 6
		bulge_frac = 4
	return disk_frac,bulge_frac

def spirals(h_d):			
	func = 'log'
	rad_in = 0.
	rad_out = h_d/2.
	return func,rad_in,rad_out

