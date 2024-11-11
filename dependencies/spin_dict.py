import numpy as np
from itertools import combinations

import dependencies.estruct_interface as ei
import dependencies.constants as const
import dependencies.vector_funcs as vf

#from dependencies.estruct_interface import seperate_string_number
#from dependencies.constants import gyro_dict, spin_type_dict, ang_to_m, hbar, mu_naught
#from dependencies.vector_funcs import angle_between, vec_magnitude

def spin_dict_from_xyzs(cPos,xyzs):

    '''
    Generates a dictionary of information about spins for use in the spin Hamiltonian. 
    Contains information about the xyz coordinates of the spins, as well as their magnetic moments and types.
    Assumes that the central spin is an electron. 
    Takes as input the position of the electron, and a dictionary with the spin types as keys and their xyz 
    coordinates as values, not including the electron.
    The spin types dictionary should have the nuclei numbered, e.g. '1H','2H', etc.
    Also returns the total dimension of the Spin Hamiltonian, as well as an list of the spin types in the system.
    '''
    
    sPos = [cPos]
    spin_types = []
    
    spins = {0:[cPos,const.gyro_dict['e']]}
    index = 0
    for spin,key in enumerate(list(xyzs.keys())):
        spins[index + 1] = [xyzs[key],const.gyro_dict[ei.seperate_string_number(key)[-1]]]
        spin_types.append(ei.seperate_string_number(key)[-1])
        index += 1
        
    spin_types.insert(0,'e')
    
    dim = 1
    for type_ in spin_types:
        
        dim *= 2*const.spin_type_dict[type_] + 1
        
    return int(dim), spins, spin_types

def get_imap(numSpins,spin_dict,readin_tensors):

    '''
        Generates the interactions between central and bath spins, and bath - bath spins.
        Has the option to be calculated according to two different equations. 
        pdip=True - point-dipole equation from PyCCE documentation.
        pdip=False - dipole-dipole equation from most of the other literature.
        Default is pdip=False.
        Takes as input:
        - the number of spins in the system
        - the spin dictionary.
        Optional input:
        - hyperfine tensors from ORCA output.
    '''
    
    zhat = np.array([0,0,1])
    
    intes = list(range(0,numSpins))
    interactions = combinations(intes,2)
    
    imap = {}
        
    for index,inte in enumerate(interactions):
                
        if inte[0] == 0:
            imap[f'{inte}'] = readin_tensors[inte[1]]
                    
        else:
            dif_vec = (spin_dict[inte[0]][0])*const.ang_to_m - (spin_dict[inte[1]][0])*const.ang_to_m
            # multiplying by 1e11 to convert all the s's to ms's
            bnm = -(1/4)*(1e8)*(1e3)*(const.hbar*const.mu_naught/(2*np.pi))*spin_dict[inte[0]][1]*spin_dict[inte[1]][1]*(1 - 3*np.cos(vf.angle_between(dif_vec,zhat))**2)/((vf.vec_magnitude(dif_vec))**3)
            imap[f'{inte}']=bnm
    
                
    return imap
