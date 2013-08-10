############## SImulation of GALaxies ############## 
##############     Version 1.0.1      ##############          
# Written by Mosenkov Alexander as a part of my PhD work. St. Petersburg State University, RUSSIA, 2013.
# It is absolutely for free according to the GNU license but please remember to give proper acknowledgement if you're using this code.
# You can also edit it if you want and email me about that. Let me know if you've found any mistakes in the code so I could fix them.
# My email:	mosenkovAV@gmail.com
####################################################
#!/usr/bin/python
# -*- coding:  cp1251 -*-

import sys
import math
import numpy as np
from scipy import stats
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.ticker import NullFormatter
from numpy import *
from pylab import *
import os
import shutil
import subprocess
import time
import pyfits
tmp_out = sys.stdout
import random 


#import model_input as inp
import setup

time_begin = time.time() 




#*** Colour fonts ***
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

configdir = '.'
sys.path.append(configdir)
DECA_PATH = os.path.abspath(__file__)

model_input = 'model_input.py'
import model_input as inp

par_file = setup.par_file
box_psf = setup.box_psf
window = setup.window

def MagBulge_meb (reb,MagBulge,n):
	nu = 1.9987*n-0.3267
	fn = n*(np.exp(nu))*special.gamma(2.0*n)/(nu**(2.0*n))
	meb = MagBulge + 2.5*(log10(fn) + log10(2.0*math.pi*reb*reb)) + 36.57
	return meb

def mod_psf(fwhm_psf,M_psf,ell,PA,m0,pix2secx,pix2secy):
	nx_psf = box_psf
	ny_psf = box_psf
	if nx_psf%2==0:
		xc_psf = int(nx_psf/2. + 1)
	else:
		xc_psf = int(nx_psf/2. + 0.5)
	if ny_psf%2==0:
		yc_psf = int(ny_psf/2. + 1)
	else:
		yc_psf = int(ny_psf/2. + 0.5)

	#========================================
	#pix2secx = pix2sec
	#pix2secy = pix2sec
	#========================================


	f = open(r"modelPSF.txt", "w") 
	sys.stdout = f
	print "\n==============================================================================="
	print "# IMAGE and GALFIT CONTROL PARAMETERS"
	print "A) none                # Input data image (FITS file)"
	print "B) psf.fits         # Output data image block"
	print "C) none                # Sigma image name (made from data if blank or none)" 
	print "D) none                # Input PSF image and (optional) diffusion kernel"
	print "E) 1                   # PSF fine sampling factor relative to data" 
	print "F) none                # Bad pixel mask (FITS image or ASCII coord list)"
	print "G) none                # File with parameter constraints (ASCII file)" 
	print "H) 1    %i   1    %i   # Image region to fit (xmin xmax ymin ymax)" % (nx_psf, ny_psf)
	print "I) %.3f    %.3f        # Size of the convolution box (x y)" % (0, 0)
	print "J) %.3f              # Magnitude photometric zeropoint" % (m0)
	print "K) %.3f    %.3f        # Plate scale (dx dy)    [arcsec per pixel]" % (pix2secx,pix2secy)
	print "O) regular             # Display type (regular, curses, both)"
	print "P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n"

	print "# INITIAL FITTING PARAMETERS"
	print "#"
	print "#   For object type, the allowed functions are:" 
	print "#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat," 
	print "#       ferrer, powsersic, sky, and isophote." 
	print "#"  
	print "#   Hidden parameters will only appear when they're specified:"
	print "#       C0 (diskyness/boxyness)," 
	print "#       Fn (n=integer, Azimuthal Fourier Modes),"
	print "#       R0-R10 (PA rotation, for creating spiral structures)."
	print "#" 
	print "# -----------------------------------------------------------------------------"
	print "#   par)    par value(s)    fit toggle(s)    # parameter description" 
	print "# -----------------------------------------------------------------------------\n"

	if window=='gauss':
		print "# Gaussian function\n"
		print "0) gaussian           # object type"
		print "1) %.3f  %.3f  0 0   # position x, y        [pixel]" % (xc_psf,yc_psf)
		print "3) %.3f       0        # total magnitude" % (M_psf)     
		print "4) %.3f       0        #   FWHM               [pixels]" % (fwhm_psf)
		print "9) %.3f        0       # axis ratio (b/a)" % (1.-ell)  
		print "10) %.3f         0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA)
		print "Z) 0                  # leave in [1] or subtract [0] this comp from data"
		print "\n================================================================================"

	if window=='moffat':
		print "# Moffat function\n"
		print "0) moffat           # object type"
		print "1) %.3f  %.3f  0 0   # position x, y        [pixel]" % (xc_psf,yc_psf)
		print "3) %.3f       0        # total magnitude" % (M_psf)     
		print "4) %.3f       0        #   FWHM               [pixels]" % (fwhm_psf)
		print "5) %.3f        0       # powerlaw" % (beta)
		print "9) %.3f        0       # axis ratio (b/a)" % (1.-ell)  
		print "10) %.3f         0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA)
		print "Z) 0                  # leave in [1] or subtract [0] this comp from data"
		print "\n================================================================================"          

	sys.stdout = tmp_out
	f.close()
	os.chmod(r"modelPSF.txt",0777)
	subprocess.call("galfit modelPSF.txt", shell=True)


# Characteristics of the image: size, psf, noise, sky, contaminents, 
def createNoise(nPoints, background,gain,readnoise,exptime,ncombine):
    """
    Create an image with Poisson + Gaussian noise corresponding
    to the sky background and the exposure time
    """
    import numpy
    from numpy import random

    absBackground = background*gain	# now in electrons

    # generate the Poisson and Gaussian signal
    error=0
    while error==0:
	try:
    		ran_arr = random.poisson(absBackground, nPoints) + ncombine*random.normal(0., readnoise, nPoints)
		#ran_arr = absBackground + ncombine*random.normal(0., readnoise, nPoints)
		error=1
	except:
		error=0


    float_arr = numpy.array(ran_arr/gain, dtype=numpy.float32)

    return float_arr


#1 HEADER:
def header(file_image,file_psf,nx,ny,nx_psf,ny_psf,m0,pix2secx,pix2secy):
	if file_psf=='N0':
		nx_psf=0
		ny_psf=0
	
	print "\n==============================================================================="
	print "# IMAGE and GALFIT CONTROL PARAMETERS"
	print "A) none                # Input data image (FITS file)"
	print "B) %s  	              # Output data image block" % (file_image)
	print "C) none                # Sigma image name (made from data if blank or none)" 
	print "D) %s                  # Input PSF image and (optional) diffusion kernel" % (file_psf)
	print "E) 1                   # PSF fine sampling factor relative to data" 
	print "F) none                # Bad pixel mask (FITS image or ASCII coord list)"
	print "G) none                # File with parameter constraints (ASCII file)" 
	print "H) 1    %i   1    %i   # Image region to fit (xmin xmax ymin ymax)" % (nx, ny)
	print "I) %i    %i        # Size of the convolution box (x y)" % (nx_psf+1, ny_psf+1)
	print "J) %.3f              # Magnitude photometric zeropoint" % (m0)
	print "K) %.3f    %.3f        # Plate scale (dx dy)    [arcsec per pixel]" % (pix2secx,pix2secy)
	print "O) regular             # Display type (regular, curses, both)"
	print "P) 1                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n"

	print "# INITIAL FITTING PARAMETERS"
	print "#"
	print "#   For object type, the allowed functions are:" 
	print "#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat," 
	print "#       ferrer, powsersic, sky, and isophote." 
	print "#"  
	print "#   Hidden parameters will only appear when they're specified:"
	print "#       C0 (diskyness/boxyness)," 
	print "#       Fn (n=integer, Azimuthal Fourier Modes),"
	print "#       R0-R10 (PA rotation, for creating spiral structures)."
	print "#" 
	print "# -----------------------------------------------------------------------------"
	print "#   par)    par value(s)    fit toggle(s)    # parameter description" 
	print "# -----------------------------------------------------------------------------\n"

# Add components to the main object:
def DISK_FO (xc_d,yc_d,m0_d,h_d,PA_d,bend_mode,ampl,incl,PA,rmax_d=0.,ell_d=0.,c_d=0.,number_disk1=0,number_disk2=0,r_br=0.):
	  print "# Exponential disk function\n"
 	  print "0) expdisk1            # Object type"
 	  print "1) %.3f  %.3f  0 0     # position x, y        [pixel]" % (xc_d,yc_d)
 	  print "3) %.3f       0       # total magnitude" % (m0_d)
	  print "4) %.3f       0       # disk scale-length    [Pixels]" % (h_d)
	  print "9) %.3f       0       # axis ratio (b/a)" % (1.-ell_d)
   	  print "10) %.3f       0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA_d)
 	  print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)"
	  #if bend_mode!=99999. and ampl!=99999.:
	  #	  if bend_mode==2:
	  #		print "B2) %.3f  0      # Bending mode 2 (banana shape)" % (ampl)
	  #	  if bend_mode==3:
	  #		print "B3) %.3f  0      # Bending mode 3 (S-shape)" % (ampl)
	  #if c_d>0.:
	  #	print "C0) %.3f       1       # diskiness/boxiness" % (c_d)


	  '''
	  if psi!=99999. and gamma!=99999.:
		print "R0) %s                # PA rotation func. (tanh, sqrt, log, linear, none)" % (func)
		print "R1) %.3f        0       # bar radius  [pixels]" % (rad_in)
		print "R2) %.3f       0       # 96 asymptotic radius (i.e. at 96 of tanh rotation)" % (rad_out)
		print "R3) %.3f   0       # cumul. coord. rotation out to asymp. radius [degrees]" % (psi)
		print "R4) %.3f	       0       # Logarithmic winding scale radius [pixels]" % (gamma)
		print "R9) %.3f   0       # Inclination to L.o.S. [degrees]" % (incl)
		print "R10) %.3f  0       # Sky position angle" % (PA)
	  '''
	  if number_disk2>0:
	  	print "To) %i                  # Outer truncation by component disk2" % (number_disk2)
	  	print "T0) radial	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (r_br)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (5.)
	  elif  number_disk1>0:
	  	print "Ti) %i                  # Inner truncation by component disk1" % (number_disk1)
	  	print "T0) radial	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (r_br)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (5.)
		if rmax_d>0.:
	  		print "To) %i                  # Outer truncation by component 2" % (number_disk1+1)
	  		print "T0) radial	 #  truncation" 
	  		print "T4) %.3f       0      #  Trancation radius-break end" % (rmax_d)
	  		print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (10.)
	  elif number_disk2==0 and number_disk1==0 and rmax_d>0.:
	  	print "To) 2                  # Outer truncation by component 2"
	  	print "T0) radial	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (rmax_d)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (10.)




def DISK_EO (xc_d,yc_d,m0_d,h_d,rmax_d,PA_d,z0,c_d=-1.,number_disk1=0,number_disk2=0,r_br=0.):
	  print "# Edge-on disk function\n"
 	  print "0) edgedisk            # Object type"
 	  print "1) %.3f  %.3f  0 0     # position x, y        [pixel]" % (xc_d,yc_d)
 	  print "3) %.3f       0       # central surface brightness  [mag/arcsec^2]" % (m0_d)
 	  print "4) %.3f       0       # disk scale-height    [Pixels]" % (z0)
	  print "5) %.3f       0       # disk scale-length    [Pixels]" % (h_d)
	  print "10) %.3f         0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA_d)
 	  print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)"
	  print "C0) %.3f       1       # diskiness/boxiness" % (c_d)
	  if number_disk2>0:
	  	print "To) %i                  # Outer truncation by component disk2" % (number_disk2)
	  	print "T0) length	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (r_br)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (3.)
	  elif  number_disk1>0:
	  	print "Ti) %i                  # Inner truncation by component disk1" % (number_disk1)
	  	print "To) %i                  # Outer truncation by component 2" % (number_disk1+1)
	  	print "T0) length	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (r_br+3)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (3.)


	  	print "T0) length	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (rmax_d)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (5.)
	  elif number_disk2==0 and number_disk1==0:
	  	print "To) 2                  # Outer truncation by component 2"
	  	print "T0) length	 #  truncation" 
	  	print "T4) %.3f       0      #  Trancation radius-break end" % (rmax_d)
	  	print "T5) %.3f      0          #  Softening length (1 normal flux) [pixels]" % (7.)

def BULGE (xc_bul,yc_bul,me_bul,re_bul,n_bul,ell_bul,PA_bul,c_bul):
	 print "\n#Sersic function\n"
	 print "0) sersic2             # Object type"
	 print "1) %.3f  %.3f  0 0    # position x, y        [pixel]" % (xc_bul,yc_bul)
	 print "3) %.3f       0       # total magnitude"  % (me_bul)   
	 print "4) %.3f       0       #     R_e              [Pixels]" % (re_bul)
	 print "5) %.3f       0       # Sersic exponent (deVauc=4, expdisk=1)" % (n_bul) 
	 print "9) %.3f       0       # axis ratio (b/a)" % (1.-ell_bul)  
	 print "10) %.3f       0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA_bul)
	 print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)"
	 print "C0) %.3f       1       # diskiness/boxiness" % (c_bul)       

def BAR (xc_bar,yc_bar,me_bar,re_bar,n_bar,ell_bar,PA_bar,c_bar,rbar1,rbar2):
	 print "# Sersic function\n"
	 print "0) sersic2             # Object type"
	 print "1) %.3f  %.3f  0 0    # position x, y        [pixel]" % (xc_bar,yc_bar)
	 print "3) %.3f       0       # total magnitude"  % (me_bar)   
	 print "4) %.3f       0       #     R_e              [Pixels]" % (re_bar)
	 print "5) %.3f       0       # Sersic exponent (deVauc=4, expdisk=1)" % (n_bar) 
	 print "9) %.3f       0       # axis ratio (b/a)" % (1.-ell_bar)  
	 print "10) %.3f       0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA_bar)
	 print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)"          
	 #print "T3) %.3f       0      #  Trancation radius-break beginning" % (rbar1)
	 print "T4) %.3f       0      #  Trancation radius-break end" % (rbar2)
	 print "C0) %.3f       0       # diskiness/boxiness" % (c_bar)

def AGN (xc_agn,yc_agn,m0_agn,fwhm_agn,ell_agn):
	 print "Gaussian function for AGN"
 	 print "0) gaussian1           # object type"
	 print "1) %.3f  %.3f  0 0  # position x, y        [pixel]" % (xc_agn,yc_agn)
	 print "3) %.3f       0       # total magnitude" % (m0_agn)     
	 print "4) %.3f       0       #   FWHM               [pixels]" % (fwhm_agn)
	 print "9) %.3f      0       # axis ratio (b/a)" % (1.-ell_agn)
	 print "10) 90.0         0       # position angle (PA)  [Degrees: Up=0, Left=90]"
	 print "Z) 1                  # leave in [1] or subtract [0] this comp from data?"

'''
def Spirals (func,rad_in,rad_out,psi,gamma,incl,PA):
	print "R0) %s                # PA rotation func. (tanh, sqrt, log, linear, none)" % (func)
	print "R1) %.3f        0       # bar radius  [pixels]" % (rad_in)
	print "R2) %.3f       0       # 96 asymptotic radius (i.e. at 96 of tanh rotation)" % (rad_out)
	print "R3) %.3f   0       # cumul. coord. rotation out to asymp. radius [degrees]" % (psi)
	print "R4) %.3f	       0       # Logarithmic winding scale radius [pixels]" % (gamma)
	print "R9) %.3f   0       # Inclination to L.o.S. [degrees]" % (0.)
	print "R10) %.3f  0       # Sky position angle" % (PA)
'''

# this function is for a disk with spirals
def Spirals(disk_x, disk_y, disk_surf_bri, disk_scale, disk_axis_ratio, disk_pa, pitch_angle):
    spirals_width = random.gauss(0.65, 0.05)
    spirals_posang = random.uniform(0, 180)
    inner_radius = random.gauss(0.75 * disk_scale, 0.1)
    outer_radius = random.gauss(2.40 * disk_scale, 0.25)
    cumul_rotation = math.degrees(math.log(outer_radius/inner_radius) / math.tan(radians(pitch_angle)))
    # set random rotation direction:
    # with positive values of cumul_rotation the ends of spirals will point in the counterclockwise direction
    cumul_rotation *= random.choice((-1, 1))
    # incl of the galaxy in degrees (see http://leda.univ-lyon1.fr/leda/param/incl.html )
    galaxy_incl = math.degrees(math.asin(math.sqrt((1-10**(-2*math.log10(1/disk_axis_ratio)))/(1-10**(-2*0.5)))))
    print "0) expdisk1               #  Component type"
    print "1) %1.3f  %1.3f      1 1  #  Position x, y" % (disk_x, disk_y)
    print "3) %1.3f     1          #  Surface brightness at Rs  [mag/arcsec^2]" % (disk_surf_bri)
    print "4) %1.3f     1          #  R_s (disk scale-length)   [pix]" % (disk_scale)
    print "9) %1.3f     1          #  Axis ratio (b/a) (spirals width actually)" % (spirals_width)
    print "10) %1.3f     1          #  Position angle (PA) [deg: Up=0, Left=90]" % (spirals_posang)    
    print "R0) log                    #  PA rotation func. (power, log, none)" # only log by now
    print "R1) %1.3f     1          #  Spiral inner radius [pixels]" % (inner_radius)
    print "R2) %1.3f     1          #  Spiral outer radius [pixels]" % (outer_radius)
    print "R3) %1.3f    1          #  Cumul. rotation out to outer radius [degrees]" % (cumul_rotation)
    print "R4) 1.0000      0          #  Logarithmic winding scale radius " # always equal 1
    print "R9) %1.3f     1          #  Inclination to L.o.S. [degrees]" % (galaxy_incl)
    print "R10) %1.3f    1          #  Sky position angle" % (disk_pa)
    print " Z) 0                      #  Skip this model in output image?  (yes=1, no=0)"
    return spirals_width,spirals_posang,inner_radius,outer_radius,cumul_rotation,galaxy_incl

def Bends (bend_mode,ampl):
	if bend_mode==2:
		print "B2) %.3f  0      # Bending mode 2 (banana shape)" % (ampl)
	if bend_mode==3:
		print "B3) %.3f  0      # Bending mode 3 (S-shape)" % (ampl)
	print"\n"

def contaminents(Nmax_stars,Nmax_gals,nx,ny,fwhm,m0):
	 if Nmax_stars>0:	Nstars = random.randint(1,Nmax_stars)
	 if Nmax_gals>0:	Ngals = random.randint(1,Nmax_gals)

	 # ***parameters***
	 Mmin_star = 12. 
	 Mmax_star = 16.
	 fwhmin_star = fwhm*1.
	 fwhmax_star = fwhm*1.

	 Mmin_gal = 12. 
	 Mmax_gal = 16.
	 remin_gal = 10.
	 remax_gal = ny/10.
	 nmin_gal = 0.5
	 nmax_gal = 8.
	 bamin_gal = 0.2
	 bamax_gal = 1.
	 PAmin_gal = 0.
	 PAmax_gal = 90.
	 #****************
	 if Nmax_stars>0:
		 # STARS
		 for i in range(Nstars):
		  xc_star = random.randint(0,nx-1)
		  yc_star = random.randint(0,ny-1)
		  M_star = random.uniform(Mmin_star,Mmax_star)
		  fwhm_star = random.uniform(fwhmin_star,fwhmax_star)

		  print "# Gaussian function\n"
		  print "0) gaussian           # object type"
		  print "1) %.3f  %.3f  0 0   # position x, y        [pixel]" % (xc_star,yc_star)
		  print "3) %.3f       0        # total magnitude" % (M_star)     
		  print "4) %.3f       0        #   FWHM               [pixels]" % (fwhm_star)
		  print "9) 1.0        0       # axis ratio (b/a)"   
		  print "10) 0.0         0       # position angle (PA)  [Degrees: Up=0, Left=90]"
		  print "Z) 0                  # leave in [1] or subtract [0] this comp from data"

	 if Nmax_gals>0:
		 # GALAXIES
		 for i in range(Ngals):
		  xc_gal = random.randint(0,nx-1)
		  yc_gal = random.randint(0,ny-1)
		  M_gal = random.uniform(Mmin_gal,Mmax_gal)
		  re_gal = random.uniform(remin_gal,remax_gal)
		  n_gal = random.uniform(nmin_gal,nmax_gal)
		  ba_gal = random.uniform(bamin_gal,bamax_gal)
		  PA_gal = random.uniform(PAmin_gal,PAmax_gal)

		  print "# Sersic function \n"
		  print "0) sersic             # Object type"
		  print "1) %.3f  %.3f  0 0     # position x, y        [pixel]" % (xc_gal,yc_gal)
		  print "3) %.3f       0       # total magnitude" % (M_gal)   
		  print "4) %.3f       0       #     R_e              [Pixels]" % (re_gal)
		  print "5) %.3f       0       # Sersic exponent (deVauc=4, expdisk=1)" % (n_gal)  
		  print "9) %.3f       0       # axis ratio (b/a)" % (ba_gal)   
		  print "10) %.3f       0       # position angle (PA)  [Degrees: Up=0, Left=90]" % (PA_gal)
		  print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)\n"

def sky(sky_level,dskyX,dskyY):
	 print "# sky\n"
 	 print "0) sky                # Object type"
 	 print "1) %.3f       0       # sky background       [ADU counts]" % (sky_level)
 	 print "2) %.3f      0       # dsky/dx (sky gradient in x)" % (dskyX)
	 print "3) %.3f      0       # dsky/dy (sky gradient in y)" % (dskyY)
	 print "Z) 0                  #  Skip this model in output image?  (yes=1, no=0)"


def noise(image_in,image_out,gain,readnoise,exptime,ncombine):
	# Adding noise
	#http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1995ApJ...448..563B&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf
	#http://arxiv.org/pdf/1205.6319v1.pdf
	#http://asp.eurasipjournals.com/content/pdf/1687-6180-2005-643143.pdf

	shutil.copy(image_in,image_out) 
	hdulist1 = pyfits.open(image_out, do_not_scale_image_data=True, mode='update')
	img = hdulist1[0].data
	(dimy,dimx) = img.shape
	nPoints = [dimy,dimx]

	img1 = createNoise(nPoints, img,gain,readnoise,exptime,ncombine)
	for i in range(dimy):
		for k in range(dimx):
			img[i,k] = img1[i,k]   #new fits-file img_new with ADU	

	hdulist1.flush()



def dust(image_in,image_out,xc,yc,h,zd,tau_f):
	k0 = tau_f/(2.*zd)
	def k(R,z):
		return k0*np.exp(-fabs(R)/hd - fabs(z)/zd)
	def tau(R,z):
		return 2.*k0*fabs(R)* sp.special.kn(1,fabs(R)/h) * ( 1. / (np.cosh(fabs(z)/zd))**2   )

	shutil.copy(image_in,image_out) 
	hdulist1 = pyfits.open(image_out, do_not_scale_image_data=True, mode='update')
	img = hdulist1[0].data
	ny,nx = np.shape(img)

	for i in range(ny):
		for k in range(nx):
			R = k - xc+0.5
			z = i - yc+0.5
			img[i,k] = img[i,k]*np.exp(-tau(R,z))
	hdulist1.flush()


def warps(image_in,image_out,xc,yc,Al,Cl,Ar,Cr,backgr,z0,rmax):
	def ff(x,A,C):
		return C*(fabs(x)-A)

	shutil.copy(image_in,image_out)

	hdulist = pyfits.open(image_in)
	img = hdulist[0].data

 	hdulist1 = pyfits.open(image_out, do_not_scale_image_data=True, mode='update')
	img_new = hdulist1[0].data
	ny,nx = np.shape(img)

	rmax = rmax*3.5	

	# Consider the left warp:
	zmax = 5.*z0 # fabs(math.sin(arctan(Cl)) * (rmax-fabs(Al)))
	x = range(int(xc)-int(rmax),int(xc)-int(Al),1)
	y = range(int(yc)-int(zmax),int(yc)+int(zmax),1)

	#print xc,yc,Al,Cl,Ar,Cr,backgr,z0,rmax
	#exit()

	#print nx,ny
	#print min(x),max(x),min(y),max(y)


	
	for i in y:
		for k in x:
			if i>=0 and i<ny-1 and k>=0 and k<nx-1:
					#if img[i,k]>1.*backgr:
					#print 'ok'
					img_new[i,k] = img[i-int(ff(k-int(xc),Al,Cl)),k]

	# Consider the right warp:
	zmax =  5.*z0 #fabs(math.sin(arctan(Cr)) * (rmax-fabs(Ar)))
	x = range(int(xc)+int(Ar),int(xc)+int(rmax),1)
	y = range(int(yc)-int(zmax),int(yc)+int(zmax),1)


	#print nx,ny
	#print min(x),max(x),min(y),max(y)

	for i in y:
		for k in x:
			if i>=0 and i<ny-1 and k>=0 and k<nx-1:
					#if img[i,k]>1.*backgr:
					img_new[i,k] = img[i-int(ff(k-int(xc),Ar,Cr)),k]
				

	hdulist1.flush()



##########################################################################################################
##########################################################################################################
##########################################################################################################

def main_model():
	if os.path.exists('login.cl')==False:
		sys.exit(bcolors.FAIL+ 'WARNING! File login.cl does not exist! Run mkiraf!'+ bcolors.ENDC)
	number = inp.number_of_galaxies
	Number = number
	gal_type = inp.gal_type
	if number<=0 or type(number)==float:
		sys.exit(bcolors.FAIL+ 'The number of modelling galaxies is incorrect!'+ bcolors.ENDC)
	if Number==1:
		number=2


	f = open(par_file, "w")
	fff = open('deca_input.dat', "w")
 
	print >>f, '#\tm0d\th\tz0\trtr\tell_d\tPA_d\tmeb\treb\tn\tell_b\tc_b\tPA_b\tPSF\tsky\tz_d\ttau_d\tpitch_angle\tincl\tModel\tSc\tsc'
	print >>f, '\tm/arc2\tkpc\tkpc\tkpc\t\t\tm/arc2\tkpc\t\t\t\t\tarc\tDN\tkpc\tkpc\tdeg\t\tkpc/arc\tarc/pix'



	for k in range(0,number,1):
		# About the image
		image_out = str(k) + '.fits'
		GAIN = random.uniform(inp.GAIN[0],inp.GAIN[1])

		readnoise = random.uniform(inp.readnoise[0],inp.readnoise[1])
		m0 = inp.m0
		ncombine = inp.ncombine
		exptime = inp.exptime
		pix2secx = inp.pix2secx
		pix2secy = inp.pix2secy
		nx = inp.nx
		ny = inp.ny
		ima_filter = inp.ima_filter
		

		#scale = inp.scale
		scale = random.uniform(inp.scale[0],inp.scale[1])

		Scale = 1./(scale*sqrt(pix2secx*pix2secy))

		use_corr = inp.use_corr


		# About the PSF
		fwhm_psf_type = inp.fwhm_psf_type
		fwhm_psf = inp.fwhm_psf
		if fwhm_psf_type=='uniform':
			fwhm = random.uniform(fwhm_psf[0],fwhm_psf[1])
		elif fwhm_psf_type=='normal':
			fwhm = random.normalvariate(fwhm_psf[0],fwhm_psf[1])
		

		# Parameters of the image and components
		if gal_type=='all':
			disk_frac,bulge_frac=setup.rel_numb(gal_type,scale)


		if (gal_type=='all' and random.randint(0,9)<disk_frac) or gal_type=='disk' or gal_type=='bulgeless':
			# The DISK
			if use_corr=='YES':
				k1,b1 = setup.disk(ima_filter)

			incl = inp.incl
			i_d = random.uniform(incl[0],incl[1])

			S0d_type = inp.S0d_type
			S0d = inp.S0d
			if S0d_type=='uniform':
				S0_d = random.uniform(S0d[0],S0d[1])
			elif S0d_type=='normal':
				S0_d = random.normalvariate(S0d[0],S0d[1])
	
			if use_corr=='YES':
				h_d = 10**( (S0_d-b1)/k1 )
				if i_d>=setup.incl_lim:
					z0_d = h_d*random.uniform(0.15,0.3)
					#S0_d = S0_d + 2.5*log10(z0_d/h_d)
				else:
					z0_d=99999.
			else:
				h_type = inp.h_type
				h = inp.h
				if h_type=='uniform':
					h_d = random.uniform(h[0],h[1])
				elif h_type=='normal':
					h_d = random.normalvariate(h[0],h[1])
				if i_d>=setup.incl_lim:
					z0_type = inp.z0_type
					z0 = inp.z0
					if z0_type=='uniform':
						z0_d = random.uniform(z0[0],z0[1])
					elif z0_type=='normal':
						z0_d = random.normalvariate(z0[0],z0[1])
					#S0_d = S0_d + 2.5*log10(z0_d/h_d)
				else:
					z0_d=99999.

			xc_d = int(nx/2.)
			yc_d = int(ny/2.)
			PA_d = 90.#random.uniform(0, 90)

			q0 = setup.q0

			rtr_type = inp.rtr_type
			rtr = inp.rtr
			if i_d>=setup.incl_lim:
				ell_d = 99999.	# >1. in case of edge-on galaxy
			else:
				ell_d = 1.-sqrt( (1.-q0**2)*(cos(radians(i_d)))**2 + q0**2)

			if rtr_type=='uniform':
				rmax_d = random.uniform(rtr[0]*h_d,rtr[1]*h_d)
			elif rtr_type=='normal':
				rmax_d = random.normalvariate(rtr[0]*h_d,rtr[1]*h_d)

		else:
			S0_d = 99999.
			h_d = 99999.
			z0_d = 99999.
			ell_d = 99999.
			rmax_d = 99999.
			PA_d = 99999.
			i_d = 99999.


		if gal_type=='all' or gal_type=='ell' or gal_type=='disk':
			lot=2
			# The BULGE
			xc_bul = int(nx/2.)
			yc_bul = int(ny/2.)


			q_bul_type = inp.q_bul_type
			qbul = inp.q_bul
			if q_bul_type=='uniform':
				q_bul = random.uniform(qbul[0],qbul[1])
			elif q_bul_type=='normal':
				q_bul = random.normalvariate(qbul[0],qbul[1])
			elif q_bul_type=='normal_bim':
				lot = random.randint(0,1)
				if lot==0:
					q_bul = random.normalvariate(qbul[0],qbul[1])
				if lot==1:
					q_bul = random.normalvariate(qbul[2],qbul[3])

			if q_bul>1.:	q_bul=1.
			if q_bul<0.45:	q_bul=0.45

			if i_d<setup.incl_lim:	q_bul = random.uniform(0.8,1.0)
				

			re_bul_type = inp.re_bul_type
			rebul = inp.re_bul

			ret = 0
			while ret == 0:
				if re_bul_type=='uniform':
					re_bul = random.uniform(rebul[0],rebul[1])
				elif re_bul_type=='normal':
					re_bul = random.normalvariate(rebul[0],rebul[1])
				elif re_bul_type=='normal_bim':
					if lot==2:
						lot = random.randint(0,1)
					if lot==0:
						re_bul = random.normalvariate(rebul[0],rebul[1])
					if lot==1:
						re_bul = random.normalvariate(rebul[2],rebul[3])
				
				n_bul_type = inp.n_bul_type
				nbul = inp.n_bul
				if n_bul_type=='uniform':
					n_bul = random.uniform(nbul[0],nbul[1])
				elif n_bul_type=='normal':
					n_bul = random.normalvariate(nbul[0],nbul[1])
				elif n_bul_type=='normal_bim':
					if lot==2:
						lot = random.randint(0,1)
					if lot==0:
						n_bul = random.normalvariate(nbul[0],nbul[1])
					if lot==1:
						n_bul = random.normalvariate(nbul[2],nbul[3])


				if use_corr=='YES':
					M_bul = setup.bulge(ima_filter,re_bul)
					if M_bul == 0.:
						ret = 0
					else:
						if n_bul>0.5:
							#print re_bul,M_bul,n_bul
							me_bul = MagBulge_meb(re_bul,M_bul,n_bul)
							ret = 1
						else:
							ret = 0				

				else:
					me_bul_type = inp.me_bul_type
					mebul = inp.me_bul
					if me_bul_type=='uniform':
						me_bul = random.uniform(mebul[0],mebul[1])
					elif me_bul_type=='normal':
						me_bul = random.normalvariate(mebul[0],mebul[1])
					elif me_bul_type=='normal_bim':
							if lot==2:
								lot = random.randint(0,1)
							if lot==0:
								me_bul = random.normalvariate(mebul[0],mebul[1])
							if lot==1:
								me_bul = random.normalvariate(mebul[2],mebul[3])
					ret = 1

			ell_bul = 1.-q_bul
			if PA_d!=99999.:	PA_bul = PA_d
			else:	PA_bul = 90.#random.uniform(0, 90)
			
			c_bul_type = inp.c_bul_type
			cbul = inp.c_bul
			if i_d>=setup.incl_lim and h_d!=99999.:
				if c_bul_type=='uniform':
					c_bul = random.uniform(cbul[0],cbul[1])
				elif c_bul_type=='normal':
					c_bul = random.normalvariate(cbul[0],cbul[1])
				elif c_bul_type=='normal_bim':
					if lot==2:
						lot = random.randint(0,1)
					if lot==0:
						c_bul = random.normalvariate(cbul[0],cbul[1])
					if lot==1:
						c_bul = random.normalvariate(cbul[2],cbul[3])
			else:
				c_bul = 0.
		elif gal_type=='bulgeless':
			re_bul = 99999.
			me_bul = 99999.
			n_bul = 99999.
			ell_bul = 99999.
			PA_bul = 99999.
			c_bul = 99999.




		# DUST

		dust_pars = inp.dust_pars
		if dust_pars[0]<=0.:
			zd = 99999.
			tau_f = 99999.
		else:
			zd = random.normalvariate(dust_pars[0],dust_pars[1])
			tau_f = random.normalvariate(dust_pars[2],dust_pars[3])


		if z0_d!=99999. and re_bul!=99999.:	model='edge+ser'
		if z0_d!=99999. and re_bul==99999.:	model='edge'
		if z0_d==99999. and re_bul!=99999. and h_d!=99999.:	model='exp+ser'
		if z0_d==99999. and re_bul==99999. and h_d!=99999.:	model='exp'
		if h_d==99999. and re_bul!=99999.:	model='ser'


		Dust = inp.dust
		if Dust=='YES' and i_d>=setup.incl_lim:
			model = model + '+dust' 

		if inp.sky_level!='NO':
			Sky_level = random.uniform(inp.sky_level[0],inp.sky_level[1])
			sky_level = 10**(0.4*(m0-Sky_level))*pix2secx*pix2secy
		else:	sky_level = 0.

		spirals =inp.spirals
		if spirals=='YES' and h_d!=99999. and i_d<setup.incl_lim:
			pitch_angle = random.uniform(inp.pitch_angle[0],inp.pitch_angle[1])
			model = model + '+spiral'
		else:
			pitch_angle = 99999.

			
		bend_mode = inp.bend_mode
		if bend_mode!='NO':
			bend_mode = random.randint(2,3)
			ampl = inp.ampl
		else:
			bend_mode = 99999.
			ampl = 99999.	

		Warps = inp.warps
		warp_type = inp.warp_type
		al = inp.Al
		cl = inp.Cl
		ar = inp.Ar
		cr = inp.Cr


		if Warps=='YES' and z0_d!=99999.:
			if warp_type=='uniform':
				Al = random.uniform(al[0]*h_d,al[1]*h_d)
				Cl = random.uniform(cl[0],cl[1])
				Ar = random.uniform(ar[0]*h_d,ar[1]*h_d)
				Cr = random.uniform(cr[0],cr[1])
			elif warp_type=='normal':
				Al = random.normalvariate(al[0]*h_d,al[1]*h_d)
				Cl = random.normalvariate(cl[0],cl[1])
				Ar = random.normalvariate(ar[0]*h_d,ar[1]*h_d)
				Cr = random.normalvariate(cr[0],cr[1])
		else:
			Al = 99999.
			Cl = 99999.
			Ar = 99999.
			Cr = 99999.					


		ima_name = './models/ima/' + str(k+1)+'.fits'
		psf_name = './models/psf/' + str(k+1)+'_psf.fits'

		if inp.compress=='gzip':
			ima_name = ima_name+'.gz'

		if inp.compress=='bzip2':
			ima_name = ima_name+'.bz2'

		print >>f, '%i\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\t%.3f\t%.3f\t%i\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (k+1,S0_d,h_d,z0_d,rmax_d,ell_d,PA_d,me_bul,re_bul,n_bul,ell_bul,c_bul,PA_bul,fwhm,sky_level,zd,tau_f,pitch_angle,i_d,model,scale,sqrt(pix2secx*pix2secy),bend_mode,ampl,Al,Cl,Ar,Cr)
		print >>fff,'%i\t%s\t%.3f\t%.3f\t%s\t%s\t%.3f\t%.3f\t%s\t%.3f\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%i\t%.3f\t%.3f\t%i\t%i\t%s\t%s\t%s\t%i\t%s\t%s\t%s\t%s\t%s\t%i' % (k+1,str(k+1),nx/2.,ny/2.,ima_filter,'g - r',0.750,0.05*scale,'NO',0.100,'NONE',m0,pix2secx,pix2secy,fwhm,GAIN,readnoise,ncombine,exptime,sky_level,1,1,'field','exp+ser','incl',2,'NO','NO',ima_name,'NONE',psf_name,k+1)

	f.close()
	fff.close()


	if Number!=1:	print bcolors.OKBLUE+ 'The sample of %i galaxies is created'% (number) + bcolors.ENDC 	
	else:	print bcolors.OKBLUE+ 'The sample of 1 galaxy is created' + bcolors.ENDC 	



	#################################################################################
	#################################################################################
	#################################################################################

	S0_d,h_d,z0_d,rmaxd,elld,PA_d,me_bul,re_bul,n_bul,ell_bul,cbul,PA_bul,fwhm,sky_level,zd,tau_f,pitch_angle,i_d,ampl,Al,Cl,Ar,Cr,scale = loadtxt(par_file, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,23,24,25,26,27,20],dtype=float, unpack=True,skiprows=2,delimiter='\t')
	number,bend_mode = loadtxt(par_file, usecols=[0,22],dtype=int, unpack=True,skiprows=2)
	model = loadtxt(par_file, usecols=[20],dtype=str, unpack=True,skiprows=2)
	Gain,Readnoise = loadtxt('deca_input.dat', usecols=[15,16],dtype=float, unpack=True,skiprows=0,delimiter='\t')

	if pitch_angle[0]!=99999.:
		f_spirals = open('spiral_par.dat', "w")
 		print >>f_spirals, '#\tpitch_angle\twidth\tPA_sp\tr_in\tr_out\tcum_rot\tincl'


	if not os.path.exists("./models/psf"):
        	os.makedirs("./models/psf")

	if not os.path.exists("./models/ima"):
        	os.makedirs("./models/ima")

	if Number==1:	N = 1
	else:	N = len(number)

	for k in range(0,N,1):
		if z0_d[k]!=99999.:
			S0_d[k] = S0_d[k] + 2.5*log10(z0_d[k]/h_d[k])
		Scale = 1./(scale[k]*sqrt(pix2secx*pix2secy))
		# Contaminents
		Nmax_stars = inp.contaminents[0]
		Nmax_gals = inp.contaminents[1]
		#sky_level = 10**(0.4*(m0-inp.sky_level))*pix2secx*pix2secy
		dskyX = setup.dskyX
		dskyY = setup.dskyY
		# Files to be created
		file_image = setup.file_image
		file_psf = setup.file_psf

		M_psf = 12.
		ell = 0.
		PA = 0.
		nx_psf = setup.nx_psf
		ny_psf = setup.ny_psf


		# PSF convolution
		if fwhm[k]<=0.:
			file_psf = 'NO'
		else:
			mod_psf(fwhm[k]/sqrt(pix2secx*pix2secy),M_psf,ell,PA,m0,pix2secx,pix2secy)
			#shutil.move('psf.fits','./models/psf/%i_psf.fits' % (number[k]))


		# Creation of the input GALFIT file
		f = open(r"modelIN.txt", "w") 
		sys.stdout = f
		header(file_image,file_psf,nx,ny,nx_psf,ny_psf,m0,pix2secx,pix2secy)

		if z0_d[k]!=99999.:
			DISK_EO (xc_d,yc_d,S0_d[k],h_d[k]*Scale,rmaxd[k]*Scale,PA_d[k],z0_d[k]*Scale,c_d=-1.,number_disk1=0,number_disk2=0,r_br=0.)
			#if bend_mode[k]>1:
			#	Bends (bend_mode[k],ampl[k])

		if elld[k]<1.:
			#func,rad_in,rad_out = setup.spirals(h_d[k])
			#teta = math.degrees(log10(gamma[k]+1.)/tan(math.radians(psi[k])))
			if pitch_angle[k]!=99999.:
				spirals_width,spirals_posang,inner_radius,outer_radius,cumul_rotation,galaxy_incl = Spirals(xc_d, yc_d, S0_d[k], h_d[k]*Scale, 1.-elld[k], PA_d[k], pitch_angle[k])
				print >>f_spirals, '%i\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%2.f\t%.2f' % (number[k],pitch_angle[k],spirals_width,spirals_posang,inner_radius,outer_radius,cumul_rotation,galaxy_incl)

			else:
				DISK_FO (xc_d,yc_d,S0_d[k],h_d[k]*Scale,PA_d[k],bend_mode[k],ampl[k],i_d[k],90.-PA_d[k],rmax_d=rmaxd[k]*Scale,ell_d=elld[k],c_d=0.,number_disk1=0,number_disk2=0,r_br=0.)

		if re_bul[k]!=99999.:	
			BULGE (xc_bul,yc_bul,me_bul[k],re_bul[k]*Scale,n_bul[k],ell_bul[k],PA_bul[k],cbul[k])


			
		sys.stdout = tmp_out
		f.close()
		os.chmod(r"modelIN.txt",0777)
		subprocess.call("galfit modelIN.txt", shell=True)

		if Warps=='YES' and z0_d[k]!=99999.:
			warps(file_image,'image_warp.fits',xc_d,yc_d,Al[k]*Scale,Cl[k],Ar[k]*Scale,Cr[k],Readnoise[k]/Gain[k],z0_d[k]*Scale,rmaxd[k]*Scale)
			shutil.move('image_warp.fits',file_image)

		if Dust=='YES' and z0_d[k]!=99999.:
			dust(file_image,'image_dust.fits',xc_d,yc_d,h_d[k]*Scale,zd[k]*z0_d[k]*Scale,tau_f[k])
			shutil.move('image_dust.fits','pure_galaxy.fits')

		else:
			shutil.move(file_image,'pure_galaxy.fits')

		os.remove('modelIN.txt')
		os.remove('modelPSF.txt')
		#os.remove(file_image)

		#exit()

		# Creation of the input GALFIT file
		f = open(r"modelIN.txt", "w") 
		sys.stdout = f
		header(file_image,file_psf,nx,ny,nx_psf,ny_psf,m0,pix2secx,pix2secy)

		contaminents(Nmax_stars,Nmax_gals,nx,ny,fwhm[k]*sqrt(pix2secx*pix2secy),m0)
		if sky_level[k]>0.:	sky(sky_level[k],dskyX,dskyY)
		sys.stdout = tmp_out
		f.close()
		os.chmod(r"modelIN.txt",0777)
		subprocess.call("galfit modelIN.txt", shell=True)
		#shutil.copy(file_image,'image_dust.fits') 
		shutil.move(file_image,'contam.fits')


		f = open("sum.cl", "w") 
		sys.stdout = f
		print "#Script: add"
		print "images"
		print "imutil"
		print "imarith pure_galaxy.fits + contam.fits %s" % (file_image)
		print "logout"	
		sys.stdout = tmp_out
		f.close()
		os.chmod(r"sum.cl",0777)
		subprocess.call("cl < sum.cl -o", shell=True)


		Noise = inp.noise

		if Noise!='NO':
			noise(file_image,image_out,Gain[k],Readnoise[k],exptime,ncombine)
			os.remove(file_image)
		else:
			shutil.move(file_image,image_out)


		if fwhm[k]>0.:	shutil.move('psf.fits','./models/psf/%i_psf.fits' % (number[k]))

		if inp.compress=='bzip2':
			print 'compressing!'
			subprocess.call("bzip2 %s" % image_out, shell=True)
			shutil.move(image_out+'.bz2','./models/ima/%i.fits.bz2' % (number[k]))
		elif inp.compress=='gzip':
			subprocess.call("gzip %s" % image_out, shell=True)
			shutil.move(image_out+'.gz','./models/ima/%i.fits.gz' % (number[k]))
		else:
			shutil.move(image_out,'./models/ima/%i.fits' % (number[k]))
		os.remove('sum.cl')
		#os.remove('modelPSF.txt')
		#os.remove(file_image)
		os.remove('contam.fits')
		os.remove('pure_galaxy.fits')
		os.remove('modelIN.txt')


		print bcolors.OKBLUE+ 'GALAXY %i is done' % (k+1) + bcolors.ENDC			

	if Number==1:
		lines = open(par_file).readlines()
		open(par_file, 'w').writelines(lines[0:3])
		#lines = open('deca_input.dat').readlines()
		#open('deca_input.dat', 'w').writelines(lines[:-1])


	#lines = file.readlines()
	#lines = lines[:-1]

	if pitch_angle[0]!=99999.:	f_spirals.close()
	time_finish = time.time() 
	duration = time_finish - time_begin
	print bcolors.OKBLUE+ 'THE END' + bcolors.ENDC
	print 'time=%.3fsec' % (duration)


main_model()
