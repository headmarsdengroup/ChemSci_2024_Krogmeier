import numpy as np
import scipy.linalg as scp
from itertools import combinations
from functools import reduce
from timeit import default_timer as timer
from scipy.optimize import curve_fit
import random
import re

#
#
### 2X2 SPIN OPERATORS - spin 1/2
#
#

sx = np.array(([[0,1/2],[1/2,0]]),dtype=np.complex128)
sy = np.array(([[0,-1/2],[1/2,0]]),dtype=np.complex128)*1j
sz = np.array(([[1/2,0],[0,-1/2]]),dtype=np.complex128)
Id = np.identity(2)

splus = np.array([[0,1],[0,0]],dtype=np.complex128)
sminus = np.array([[0,0],[1,0]],dtype=np.complex128)

#
#
### 3x3 SPIN OPERATORS - spin 1
#
#

sx3 = np.array(([[0,1,0],[1,0,1],[0,1,0]]),dtype=np.complex128)*(1/np.sqrt(2))
sy3 = np.array(([[0,-1,0],[1,0,-1],[0,1,0]]),dtype=np.complex128)*(1j/np.sqrt(2))
sz3 = np.array(([[1,0,0],[0,0,0],[0,0,-1]]),dtype=np.complex128)
Id3 = np.identity(3)

#
#
### 4x4 SPIN OPERATORS - spin 3/2, etc.
#
#

sx4 = np.array(([[0,np.sqrt(3),0,0],[np.sqrt(3),0,2,0],[0,2,0,np.sqrt(3)],[0,0,np.sqrt(3),0]]),dtype=np.complex128)*(1/2)
sy4 = np.array(([[0,-np.sqrt(3),0,0],[np.sqrt(3),0,-2,0],[0,2,0,-np.sqrt(3)],[0,0,np.sqrt(3),0]]),dtype=np.complex128)*(1j/2)
sz4 = np.array(([[3/2,0,0,0],[0,1/2,0,0],[0,0,-1/2,0],[0,0,0,-3/2]]),dtype=np.complex128)
Id4 = np.identity(4)

#
#
### 5x5 SPIN OPERATORS
#
#

sx5 = np.array(([[0,2,0,0,0],[2,0,np.sqrt(6),0,0],[0,np.sqrt(6),0,np.sqrt(6),0],[0,0,np.sqrt(6),0,2],[0,0,0,2,0]]),dtype=np.complex128)*(1/2)
sy5 = np.array(([[0,-2,0,0,0],[2,0,-np.sqrt(6),0,0],[0,np.sqrt(6),0,-np.sqrt(6),0],[0,0,np.sqrt(6),0,-2],[0,0,0,2,0]]),dtype=np.complex128)*(1j/2)
sz5 = np.array(([[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,-1,0],[0,0,0,0,-2]]),dtype=np.complex128)
Id5 = np.identity(5)

#
#
### 6x6 SPIN OPERATORS
#
#

sx6 = np.array(([[0,np.sqrt(5),0,0,0,0],[np.sqrt(5),0,np.sqrt(8),0,0,0],[0,np.sqrt(8),0,np.sqrt(9),0,0],[0,0,np.sqrt(9),0,np.sqrt(8),0],[0,0,0,np.sqrt(8),0,np.sqrt(5)],[0,0,0,0,np.sqrt(5),0]]),dtype=np.complex128)*(1/2)
sy6 = np.array(([[0,-np.sqrt(5),0,0,0,0],[np.sqrt(5),0,-np.sqrt(8),0,0,0],[0,np.sqrt(8),0,-np.sqrt(9),0,0],[0,0,np.sqrt(9),0,-np.sqrt(8),0],[0,0,0,np.sqrt(8),0,-np.sqrt(5)],[0,0,0,0,np.sqrt(5),0]]),dtype=np.complex128)*(1j/2)
sz6 = np.array(([[5/2,0,0,0,0,0],[0,3/2,0,0,0,0],[0,0,1/2,0,0,0],[0,0,0,-1/2,0,0],[0,0,0,0,-3/2,0],[0,0,0,0,0,-5/2]]),dtype=np.complex128)
Id6 = np.identity(6)

#
#
### 8x8 SPIN OPERATORS
#
#

sx8 = np.array(([[0,np.sqrt(7),0,0,0,0,0,0],[np.sqrt(7),0,np.sqrt(12),0,0,0,0,0],[0,np.sqrt(12),0,np.sqrt(15),0,0,0,0],[0,0,np.sqrt(15),0,np.sqrt(16),0,0,0],[0,0,0,np.sqrt(16),0,np.sqrt(15),0,0],[0,0,0,0,np.sqrt(15),0,np.sqrt(12),0],[0,0,0,0,0,np.sqrt(12),0,np.sqrt(7)],[0,0,0,0,0,0,np.sqrt(7),0]]),dtype=np.complex128)*(1/2)
sy8 = np.array(([[0,-np.sqrt(7),0,0,0,0,0,0],[np.sqrt(7),0,-np.sqrt(12),0,0,0,0,0],[0,np.sqrt(12),0,-np.sqrt(15),0,0,0,0],[0,0,np.sqrt(15),0,-np.sqrt(16),0,0,0],[0,0,0,np.sqrt(16),0,-np.sqrt(15),0,0],[0,0,0,0,np.sqrt(15),0,-np.sqrt(12),0],[0,0,0,0,0,np.sqrt(12),0,-np.sqrt(7)],[0,0,0,0,0,0,np.sqrt(7),0]]),dtype=np.complex128)*(1j/2)
sz8 = np.array(([[7/2,0,0,0,0,0,0,0],[0,5/2,0,0,0,0,0,0],[0,0,3/2,0,0,0,0,0],[0,0,0,1/2,0,0,0,0],[0,0,0,0,-1/2,0,0,0],[0,0,0,0,0,-3/2,0,0],[0,0,0,0,0,0,-5/2,0],[0,0,0,0,0,0,0,-7/2]]),dtype=np.complex128)
Id8 = np.identity(8)

spin_operators = {0.5:{'sx':sx,'sy':sy,'sz':sz,'Id':Id,'s+':splus,'s-':sminus},
                  1:{'sx':sx3,'sy':sy3,'sz':sz3,'Id':Id3},
                 1.5:{'sx':sx4,'sy':sy4,'sz':sz4,'Id':Id4},
                  2:{'sx':sx5,'sy':sy5,'sz':sz5,'Id':Id5},
                  2.5:{'sx':sx6,'sy':sy6,'sz':sz6,'Id':Id6},
                  3.5:{'sx':sx8,'sy':sy8,'sz':sz8,'Id':Id8}
                 }


#
#
### CONSTANTS
#
#

ang_to_m = 1e-10
khz_to_J = 6.626e-31 
planck = 6.626e-34 # J s
hbar = planck/(2*np.pi)
conver = 1000/10000
kb = 1.380649e-23 # J/K
Jev = 6.242e18
NA = 6.023e23

n_mag = 5.050783699e-27 # nuclear magneton
gv = 1.47106 
ge = -2.002319304361 
gp = 5.58569469 
Be = 13996.2451684*conver 
Bn = 7.62259328*conver 
Bp = 2.793*Bn 
Bv = 5.1464*Bn 

gyro_dict = {'e':-17608.59705,'H':26.75221824,'C':6.72828532,'V':7.0492,'N':1.93297,'Cu':7.1118} # modify this as needed
spin_type_dict = {'e':0.5,'H':0.5,'C':0.5,'V':0.5,'N':1.0,'Cu':0.5} # approximating V, Cu as spin 1/2
mm_dict = {'e':-9.2740100783e-24,'H':n_mag*2.79284739,'C':n_mag*0.7024118,'V':n_mag*5.14870574,'Cu':2.43*n_mag,'N':0.403761*n_mag}

#gyro_e = -17608.59705*2*np.pi # electron in units of rad ms-1 G-1
gyro_p = 26.75221824 # hydrogen 1
gyro_p3 = 28.53508 # hydrogen 3
gyro_c = 6.72828532 # carbon 13
gyro_v = 6.728284 # vanadium 51
gyro_n = -2.7116 # nitrogen 15
hbar_mu0_4pi = 1.05457172

mu0 = 1.25663706212e-6 # kg m s-2 A-2 
mu_naught = np.pi*4e-7 # in units of kg m s-2 A-2
#e_mm_const = -9.2740100783e-24
e_mm = np.array([0,0,-9.2740100783e-24]) # electron magnetic moment J / T
e_mm_unit = e_mm / 9.2740100783e-24
p_mm_const = 1.410606797e-26
