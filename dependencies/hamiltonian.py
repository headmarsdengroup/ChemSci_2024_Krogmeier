import numpy as np
import re
import dependencies.misc as misc
import dependencies.constants as const

def H_hf(imap,dim,spin_types):
    '''
        Generates the Hamiltonian due to electron-nuclear interactions, and nuclear-nuclear interactions by
        using the interactin map object.
        Takes as input:
        - the interaction map object
        - the total dimension of the spin Hilbert space.
        - a list of the spin types in the system.
    '''
    
    sum_int_ham = np.zeros([dim,dim],dtype=np.complex128)
    for key in imap.keys():
        nums = re.findall(r'\d+', key)
        i = int(nums[0])
        j = int(nums[1])
        if imap[key].shape == (3,3):
            svec_ = misc.svec(i,spin_types)
            ivec = misc.svec(j,spin_types)
            A = imap[key]
            sum_int_ham += misc.compute_SAI(svec_,ivec,A,dim,spin_types)
        elif imap[key].shape != (3,3):
            Iiz = misc.tensor_chain(spin_types,i,'sz')
            Ijz = misc.tensor_chain(spin_types,j,'sz')
            Iip = misc.tensor_chain(spin_types,i,'s+')
            Ijm = misc.tensor_chain(spin_types,j,'s-')
            Iim = misc.tensor_chain(spin_types,i,'s-')
            Ijp = misc.tensor_chain(spin_types,j,'s+')
            A = imap[key]
            sum_int_ham += -4*(A*Iiz@Ijz) + (A*(Iip@Ijm + Iim@Ijp))

    return sum_int_ham

def H_mag(dim,numSpins,spin_dict,Bext,spin_types):
    
    '''
    Calculates the diagonal terms for the Hamiltonian, or the Hamiltonian terms that encode
    the energy of interaction with the external magnetic field.
    Takes as input:
    - dimension of the spin Hilbert space.
    - number of spins in the system.
    - spin dictionary object
    - strength of the external magnetic field (just a scalar value in the z direction).
    - list of spin types in the system.
    '''

    spin_indices = list(spin_dict.keys())
    mag_total = np.zeros([dim,dim],dtype=np.complex128)
    for spin in spin_indices:
        B = Bext
        if spin == 0:
            mag_total += spin_dict[spin][1]*B*misc.tensor_chain(spin_types,spin,'sz')
        elif spin != 0:
            mag_total += spin_dict[spin][1]*B*misc.tensor_chain(spin_types,spin,'sz')
            
    return mag_total

def H_total(hf_H,mag_H=None,fc_H=None):
    
    '''
    Just sums the Hamiltonian terms. The Fermi contact Hamiltonian and magnetic terms are not required. Kind of useless.
    '''
    
    if mag_H is None and fc_H is None:
        return hf_H
    elif mag_H is None and fc_H is not None:
        return hf_H + fc_H
    elif mag_H is not None and fc_H is None:
        return hf_H + mag_H
    else:
        return mag_H + hf_H + fc_H
